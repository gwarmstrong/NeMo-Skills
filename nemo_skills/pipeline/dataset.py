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

This module adds cluster support (SSH downloads, mount path resolution)
on top of nemo_skills.dataset.utils. The pipeline layer ONLY handles
cluster I/O — all local import/resolution logic lives in core.

For local-only dataset loading, use nemo_skills.dataset.utils directly.
"""

import importlib
import os
import tempfile
from pathlib import Path

from nemo_skills.dataset.utils import (
    ExtraDatasetType,
    add_to_path,
    import_from_path,
)
from nemo_skills.dataset.utils import (
    get_dataset_module as _get_local_dataset_module,
)
from nemo_skills.pipeline.utils import cluster_download_file, get_unmounted_path


def _get_dataset_module_from_cluster(cluster_config, mounted_path):
    """Download and import a dataset module from a remote cluster."""
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


def _get_cluster_dataset_module(dataset, data_dir, cluster_config):
    """Load dataset module when running on a cluster.

    Handles cluster-specific cases: local executor with mount paths,
    and remote executor with SSH downloads.
    """
    if data_dir is None:
        data_path = "/nemo_run/code/nemo_skills/dataset"
        dataset_module = importlib.import_module(f"nemo_skills.dataset.{dataset}")
        return dataset_module, data_path, False

    data_path = data_dir
    if cluster_config["executor"] == "local":
        with add_to_path(get_unmounted_path(cluster_config, data_dir)):
            dataset_module = importlib.import_module(dataset)
        return dataset_module, data_path, False

    dataset = dataset.replace(".", "/")
    dataset_module = _get_dataset_module_from_cluster(cluster_config, f"{data_dir}/{dataset}/__init__.py")
    return dataset_module, data_path, True


def get_dataset_module(dataset, data_dir=None, cluster_config=None, extra_datasets=None, extra_datasets_type=None):
    """Load dataset module with optional cluster support.

    This is the cluster-aware version of nemo_skills.dataset.utils.get_dataset_module.
    If cluster_config is None or executor is "none", delegates entirely to core.

    Search priority:
    1. data_dir (or ``nemo_skills.dataset`` if None) folder
    2. extra_datasets parameter if defined
    3. ``NEMO_SKILLS_EXTRA_DATASETS`` environment variable
    """
    if cluster_config is None or cluster_config["executor"] in (None, "none"):
        # No cluster — delegate entirely to core
        return _get_local_dataset_module(
            dataset, data_dir, extra_datasets=extra_datasets, extra_datasets_type=extra_datasets_type
        )

    # Cluster path: try primary location first
    try:
        return _get_cluster_dataset_module(dataset, data_dir, cluster_config)
    except ModuleNotFoundError:
        pass

    # Fallback to extra_datasets
    extra_datasets = extra_datasets or os.environ.get("NEMO_SKILLS_EXTRA_DATASETS")
    if extra_datasets is None:
        raise RuntimeError(f"Dataset {dataset} not found in {data_dir if data_dir else 'nemo_skills.dataset'}")

    if extra_datasets_type == ExtraDatasetType.cluster:
        # Cluster extra_datasets: download from remote
        dataset = dataset.replace(".", "/")
        dataset_module = _get_dataset_module_from_cluster(cluster_config, f"{extra_datasets}/{dataset}/__init__.py")
        return dataset_module, extra_datasets, True

    # Local extra_datasets: delegate to core (handles add_to_path + import)
    try:
        return _get_local_dataset_module(dataset, data_dir=extra_datasets)
    except RuntimeError:
        raise RuntimeError(
            f"Dataset {dataset} not found in any of the searched locations: "
            f"{data_dir if data_dir else 'nemo_skills.dataset'}, {extra_datasets}"
        )
