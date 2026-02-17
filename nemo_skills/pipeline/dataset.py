# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cluster-aware dataset loading.

This module adds two capabilities on top of nemo_skills.dataset.utils:
1. Mount-path resolution for local executors (get_unmounted_path)
2. SSH download for remote executors (cluster_download_file)

All import/module-resolution logic lives in core. Pipeline never calls
importlib.import_module or add_to_path directly.

For local-only dataset loading, use nemo_skills.dataset.utils directly.
"""

import os
import tempfile
from pathlib import Path

from nemo_skills.dataset.utils import (
    ExtraDatasetType,
    import_from_path,
)
from nemo_skills.dataset.utils import (
    get_dataset_module as _get_local_dataset_module,
)
from nemo_skills.pipeline.utils import cluster_download_file, get_unmounted_path


def _download_and_import(cluster_config, mounted_path):
    """Download a dataset module from a remote cluster via SSH and import it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = str(Path(tmpdir) / "init.py")
        cluster_dataset_path = get_unmounted_path(cluster_config, mounted_path)
        try:
            cluster_download_file(cluster_config, cluster_dataset_path, tmp_path)
        except FileNotFoundError:
            raise RuntimeError(
                f"Init file {mounted_path} not found on the cluster. "
                f"Please check the dataset name you're using. Did you forget to run prepare data commands?"
            )
        return import_from_path(tmp_path)


def get_dataset_module(dataset, data_dir=None, cluster_config=None, extra_datasets=None, extra_datasets_type=None):
    """Load dataset module with optional cluster support.

    This is the cluster-aware version of nemo_skills.dataset.utils.get_dataset_module.

    - No cluster / none executor: delegates entirely to core.
    - Local executor: unmounts paths, then delegates to core.
    - Remote executor: downloads from cluster via SSH when needed.
    """
    if cluster_config is None or cluster_config["executor"] in (None, "none"):
        return _get_local_dataset_module(
            dataset, data_dir, extra_datasets=extra_datasets, extra_datasets_type=extra_datasets_type
        )

    if cluster_config["executor"] == "local":
        return _get_local_executor_dataset(dataset, data_dir, cluster_config, extra_datasets, extra_datasets_type)

    return _get_remote_executor_dataset(dataset, data_dir, cluster_config, extra_datasets, extra_datasets_type)


def _get_local_executor_dataset(dataset, data_dir, cluster_config, extra_datasets, extra_datasets_type):
    """Local executor: unmount paths, then delegate entirely to core."""
    local_data_dir = get_unmounted_path(cluster_config, data_dir) if data_dir else None
    local_extra = get_unmounted_path(cluster_config, extra_datasets) if extra_datasets else None

    dataset_module, resolved_path, _ = _get_local_dataset_module(
        dataset,
        local_data_dir,
        extra_datasets=local_extra,
        extra_datasets_type=extra_datasets_type,
    )

    # Core returns the unmounted path — map it back to the mounted path
    # so pipeline callers get paths valid inside the container.
    if local_data_dir and resolved_path == local_data_dir:
        return dataset_module, data_dir, False
    if local_extra and resolved_path == local_extra:
        return dataset_module, extra_datasets, False
    return dataset_module, resolved_path, False


def _get_remote_executor_dataset(dataset, data_dir, cluster_config, extra_datasets, extra_datasets_type):
    """Remote executor: download from cluster via SSH when needed.

    Standard built-in datasets (data_dir=None) are available locally
    even on a remote executor since they ship with the code package.
    Only custom data_dir / cluster extra_datasets need SSH download.
    """
    # Try primary location
    if data_dir is not None:
        # Custom data_dir on remote cluster — must download
        try:
            dataset_name = dataset.replace(".", "/")
            dataset_module = _download_and_import(cluster_config, f"{data_dir}/{dataset_name}/__init__.py")
            return dataset_module, data_dir, True
        except RuntimeError:
            pass
    else:
        # Standard dataset — import locally via core (no cluster-type extra_datasets here,
        # that's handled in the fallback below)
        local_extra = extra_datasets if extra_datasets_type != ExtraDatasetType.cluster else None
        try:
            return _get_local_dataset_module(dataset, extra_datasets=local_extra)
        except RuntimeError:
            pass

    # Fallback to extra_datasets
    extra_datasets = extra_datasets or os.environ.get("NEMO_SKILLS_EXTRA_DATASETS")
    if extra_datasets is None:
        raise RuntimeError(f"Dataset {dataset} not found in {data_dir if data_dir else 'nemo_skills.dataset'}")

    if extra_datasets_type == ExtraDatasetType.cluster:
        dataset_name = dataset.replace(".", "/")
        dataset_module = _download_and_import(cluster_config, f"{extra_datasets}/{dataset_name}/__init__.py")
        return dataset_module, extra_datasets, True

    # Local extra_datasets on remote executor — delegate to core
    try:
        dataset_module, _, _ = _get_local_dataset_module(dataset, data_dir=extra_datasets)
        return dataset_module, extra_datasets, False
    except RuntimeError:
        raise RuntimeError(
            f"Dataset {dataset} not found in any of the searched locations: "
            f"{data_dir if data_dir else 'nemo_skills.dataset'}, {extra_datasets}"
        )
