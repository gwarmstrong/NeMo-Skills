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

Adds mount-path resolution and SSH download on top of core's
nemo_skills.dataset.utils. All import logic lives in core.
"""

import os
import tempfile
from pathlib import Path

from nemo_skills.dataset.utils import ExtraDatasetType, import_from_path
from nemo_skills.dataset.utils import get_dataset_module as _get_local_dataset_module
from nemo_skills.pipeline.utils import cluster_download_file, get_unmounted_path


def _download_and_import(cluster_config, base_path, dataset):
    """SSH-download a dataset __init__.py from the cluster and import it."""
    dataset_path = dataset.replace(".", "/")
    mounted_init = f"{base_path}/{dataset_path}/__init__.py"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = str(Path(tmpdir) / "init.py")
        try:
            cluster_download_file(cluster_config, get_unmounted_path(cluster_config, mounted_init), tmp_path)
        except FileNotFoundError:
            raise RuntimeError(
                f"Dataset {dataset} not found at {mounted_init} on the cluster. "
                "Did you forget to run prepare data commands?"
            )
        return import_from_path(tmp_path)


def get_dataset_module(dataset, data_dir=None, cluster_config=None, extra_datasets=None, extra_datasets_type=None):
    """Cluster-aware dataset loading. Delegates to core for all import logic."""
    # No cluster — pass through to core
    if cluster_config is None or cluster_config["executor"] in (None, "none"):
        return _get_local_dataset_module(
            dataset, data_dir, extra_datasets=extra_datasets, extra_datasets_type=extra_datasets_type
        )

    is_remote = cluster_config["executor"] not in ("local", None, "none")
    extra = extra_datasets or os.environ.get("NEMO_SKILLS_EXTRA_DATASETS")

    # Remote + paths on the cluster: SSH download
    if is_remote and data_dir:
        return _download_and_import(cluster_config, data_dir, dataset), data_dir, True

    if is_remote and extra_datasets_type == ExtraDatasetType.cluster and extra:
        try:
            return _get_local_dataset_module(dataset)
        except RuntimeError:
            return _download_and_import(cluster_config, extra, dataset), extra, True

    # Remote + local paths: delegate to core as-is
    if is_remote:
        return _get_local_dataset_module(
            dataset, data_dir, extra_datasets=extra, extra_datasets_type=extra_datasets_type
        )

    # Local executor: unmount paths, delegate to core, remap returned path
    local_dir = get_unmounted_path(cluster_config, data_dir) if data_dir else None
    local_extra = get_unmounted_path(cluster_config, extra) if extra else None
    dataset_module, resolved_path, _ = _get_local_dataset_module(
        dataset,
        local_dir,
        extra_datasets=local_extra,
        extra_datasets_type=extra_datasets_type,
    )
    mounted = {local_dir: data_dir, local_extra: extra}.get(resolved_path, resolved_path)
    return dataset_module, mounted, False
