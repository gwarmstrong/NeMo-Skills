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

This module extends nemo_skills.dataset.utils with cluster support
(SSH tunnels, mount path resolution, remote downloads).

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


def _get_default_dataset_module(dataset, data_dir=None, cluster_config=None):
    """Load dataset module with cluster support.

    For local-only loading (no cluster_config), delegates to
    nemo_skills.dataset.utils.get_dataset_module.
    """
    if cluster_config is None or cluster_config["executor"] in (None, "none"):
        # Delegate to core for local-only loading
        return _get_local_dataset_module(dataset, data_dir)

    is_on_cluster = False
    if data_dir is None:
        data_path = "/nemo_run/code/nemo_skills/dataset"
        dataset_module = importlib.import_module(f"nemo_skills.dataset.{dataset}")
    else:
        data_path = data_dir
        if cluster_config["executor"] == "local":
            with add_to_path(get_unmounted_path(cluster_config, data_dir)):
                dataset_module = importlib.import_module(dataset)
        else:
            dataset = dataset.replace(".", "/")
            dataset_module = _get_dataset_module_from_cluster(cluster_config, f"{data_dir}/{dataset}/__init__.py")
            is_on_cluster = True
    return dataset_module, data_path, is_on_cluster


def get_dataset_module(dataset, data_dir=None, cluster_config=None, extra_datasets=None, extra_datasets_type=None):
    """Load dataset module with optional cluster support.

    This is the cluster-aware version of nemo_skills.dataset.utils.get_dataset_module.
    If cluster_config is None or executor is "none", delegates to the core version.

    Search priority:
    1. data_dir (or `nemo_skills.dataset` if None) folder
    2. extra_datasets parameter if defined
    3. `NEMO_SKILLS_EXTRA_DATASETS` environment variable
    """
    try:
        dataset_module, data_path, is_on_cluster = _get_default_dataset_module(dataset, data_dir, cluster_config)
    except ModuleNotFoundError:
        try:
            dataset = dataset.replace(".", "/")
            extra_datasets = extra_datasets or os.environ.get("NEMO_SKILLS_EXTRA_DATASETS")
            is_on_cluster = False
            data_path = extra_datasets
            if extra_datasets is None:
                raise RuntimeError(f"Dataset {dataset} not found in {data_dir if data_dir else 'nemo_skills.dataset'}")
            if extra_datasets_type == ExtraDatasetType.local or extra_datasets_type is None:
                with add_to_path(extra_datasets):
                    dataset_module = importlib.import_module(dataset)
            else:
                dataset_module = _get_dataset_module_from_cluster(
                    cluster_config, f"{extra_datasets}/{dataset}/__init__.py"
                )
                is_on_cluster = True
        except ModuleNotFoundError:
            raise RuntimeError(
                f"Dataset {dataset} not found in any of the searched locations: "
                f"{data_dir if data_dir else 'nemo_skills.dataset'}, {extra_datasets}"
            )
    return dataset_module, data_path, is_on_cluster
