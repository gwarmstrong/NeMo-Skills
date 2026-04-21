# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""In-process vLLM launcher with the DP-on-Ray placement-group patch.

Identical interface to ``serve_vllm.py`` but starts vLLM's OpenAI API server
in the current Python process (not via subprocess), so the monkey-patches
applied before ``run_server()`` take effect inside the Ray DP coordination
path.

Motivation
----------
vLLM's ``CoreEngineActorManager.create_dp_placement_groups`` auto-discovers
every node registered with the Ray cluster and creates one placement group
per node, then asserts ``len(pgs) == dp_size``. That fails whenever a Ray
cluster has more nodes than DP replicas — for example, a 5-node allocation
running DP=4 with the 5th node reserved for a neural-eval Ray task.

NeMo Gym's ``LocalVLLMModel`` already carries the fix as two in-process
monkey-patches (``_patch_create_dp_placement_groups`` and
``_patch_init_data_parallel``). This wrapper ports them so pipelines that
serve vLLM via ``nemo_skills.inference.server`` (``nemo_gym_rollouts``,
``generate``, ``eval``, ...) can use the extra-node topology too.

The patches are intended to be temporary — see the upstream vLLM PR
discussion for the long-term fix.

Usage
-----
Drop-in replacement for ``serve_vllm.py``. Select via
``server_type=vllm_dp_ray`` when configuring a NeMo Skills pipeline.
Accepts the same ``--model``, ``--num_gpus``, ``--num_nodes``, ``--port``
arguments plus a passthrough of any extra vLLM args.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import signal
import sys
from typing import Sequence


def _apply_vllm_patches() -> None:
    """Monkey-patch vLLM's DP placement-group + DP-init logic.

    Mirrors ``LocalVLLMModelActor._patch_create_dp_placement_groups`` and
    ``_patch_init_data_parallel`` from NeMo Gym, adapted for the in-process
    serve path (no Ray actor wrapper).

    Must be called AFTER ``ray.init(...)`` connects to the cluster and
    AFTER the head placement group has been reserved.
    """
    import ray
    from ray.util.placement_group import PlacementGroup
    from vllm.v1.engine.core import DPEngineCoreProc
    from vllm.v1.engine.core import logger as dp_logger
    from vllm.v1.engine.utils import (
        CoreEngineActorManager,
        current_platform,
        envs,
        logger,
    )

    head_pg: PlacementGroup = _HEAD_PG_REF["pg"]
    server_name: str = _HEAD_PG_REF["server_name"]

    # --- create_dp_placement_groups ------------------------------------------------

    def new_create_dp_placement_groups(vllm_config):
        from ray._private.state import available_resources_per_node, total_resources_per_node

        logger.info("Creating placement groups for data parallel (patched)")
        dp_master_ip = vllm_config.parallel_config.data_parallel_master_ip
        dp_size = vllm_config.parallel_config.data_parallel_size
        dp_size_local = vllm_config.parallel_config.data_parallel_size_local

        available_resources = available_resources_per_node()

        # Strip per-PG synthetic resource keys (e.g. "node:<ip>_group_<hex>")
        # that Ray adds once placement groups exist, so the assertion
        # `len(node_ip_keys) == 1` downstream stays valid.
        for node_hex_id, node_resources in list(available_resources.items()):
            available_resources[node_hex_id] = {
                resource_id: resource
                for resource_id, resource in node_resources.items()
                if "_group_" not in resource_id
            }

        world_size = vllm_config.parallel_config.world_size

        # Start with the pre-reserved head PG (DP rank 0) instead of
        # auto-discovering every Ray node.
        placement_groups: list[PlacementGroup] = [head_pg]
        local_dp_ranks: list[int] = [0]

        dp_master_ip_key = f"node:{dp_master_ip}"
        nodes = sorted(available_resources.values(), key=lambda x: dp_master_ip_key not in x)
        assert len(nodes) > 0, "No nodes with resources found in Ray cluster."
        assert dp_master_ip_key in nodes[0], f"The DP master node (ip: {dp_master_ip}) is missing or dead"
        device_str = current_platform.ray_device_key

        n_node_devices: list[int] = [
            int(node_resources[device_str]) for node_resources in nodes if device_str in node_resources
        ]

        if dp_size == 1:
            total_nodes = total_resources_per_node().values()
            total_n_node_devices: list[int] = [
                int(node_resources[device_str]) for node_resources in total_nodes if device_str in node_resources
            ]
            max_device_per_node = max(total_n_node_devices)
        else:
            assert n_node_devices, f"No {device_str} found in Ray cluster."
            max_device_per_node = max(n_node_devices)

        pack_strategy = envs.VLLM_RAY_DP_PACK_STRATEGY
        _supported_pack_strategies = ("strict", "fill", "span")
        if pack_strategy not in _supported_pack_strategies:
            raise ValueError(
                f"{envs.VLLM_RAY_DP_PACK_STRATEGY} is not supported. "
                f"Set VLLM_RAY_DP_PACK_STRATEGY to one of {_supported_pack_strategies}"
            )

        all2all_backend = vllm_config.parallel_config.all2all_backend
        if pack_strategy == "fill" and (
            all2all_backend == "deepep_high_throughput" or all2all_backend == "deepep_low_latency"
        ):
            raise ValueError(
                "DeepEP kernels require EP ranks on the same node, "
                "which VLLM_RAY_DP_PACK_STRATEGY=fill does not guarantee. "
                "Use VLLM_RAY_DP_PACK_STRATEGY=strict instead."
            )

        if pack_strategy in ("strict", "fill"):
            placement_strategy = "STRICT_PACK"
        else:
            placement_strategy = "PACK"
            assert world_size > max_device_per_node, (
                f"World size {world_size} smaller than max devices per node "
                f"{max_device_per_node}. Set VLLM_RAY_DP_PACK_STRATEGY to "
                "'strict' or 'fill'."
            )
            if dp_size == 1:
                assert set(total_n_node_devices) == {max_device_per_node}, f"Nodes are not homogenous, {nodes}"
            else:
                assert set(n_node_devices) == {max_device_per_node}, f"Nodes are not homogenous, {nodes}"
            assert world_size % max_device_per_node == 0, (
                f"For multi-node DP, world_size ({world_size}) must be a "
                f"multiple of devices per node ({max_device_per_node})."
            )
            # Head PG already covers one DP rank's worth of compute, so
            # the remaining check is against (dp_size - 1) groups.
            assert len(n_node_devices) * max_device_per_node >= world_size * (dp_size - 1), (
                f"Not enough available nodes ({len(n_node_devices)}) and "
                f"devices per node ({max_device_per_node}) to satisfy "
                f"world_size={world_size} and dp_size={dp_size}"
            )
            assert dp_size_local == 1, (
                f"data-parallel-size-local {dp_size_local} should be 1 with VLLM_RAY_DP_PACK_STRATEGY=span."
            )

        for _ in range(dp_size - 1):
            bundles = [{device_str: 1.0}] * world_size + [{"CPU": 1.0}]
            pg_name = f"{server_name}_dp_rank_{len(placement_groups)}"
            pg = ray.util.placement_group(
                name=pg_name,
                strategy=placement_strategy,
                bundles=bundles,
            )
            placement_groups.append(pg)
            local_dp_ranks.append(0)

        if len(placement_groups) < dp_size:
            raise ValueError(
                f"Not enough resources to allocate {dp_size} placement groups, "
                f"only created {len(placement_groups)}. "
                f"Available resources: {available_resources}"
            )
        assert len(placement_groups) == dp_size, (
            f"Created {len(placement_groups)} DP placement groups, expected {dp_size}"
        )
        assert len(local_dp_ranks) == dp_size, (
            f"local_dp_ranks length {len(local_dp_ranks)} does not match expected {dp_size}"
        )

        return placement_groups, local_dp_ranks

    CoreEngineActorManager.create_dp_placement_groups = new_create_dp_placement_groups

    # --- DPEngineCoreProc._init_data_parallel --------------------------------------

    def new_init_data_parallel(self, vllm_config):
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        dp_size = vllm_config.parallel_config.data_parallel_size
        local_dp_rank = vllm_config.parallel_config.data_parallel_rank_local

        # Original vLLM: `assert dp_size > 1`. Our head-PG path makes
        # DP=1 a valid configuration (single replica, still Ray-coordinated).
        assert local_dp_rank is not None
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        if vllm_config.kv_transfer_config is not None:
            vllm_config.kv_transfer_config.engine_id = f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
            dp_logger.debug(
                "Setting kv_transfer_config.engine_id to %s",
                vllm_config.kv_transfer_config.engine_id,
            )

        self.dp_rank = dp_rank
        self.dp_group = vllm_config.parallel_config.stateless_init_dp_group()

    DPEngineCoreProc._init_data_parallel = new_init_data_parallel


# Populated by ``main()`` before the patches are applied.
_HEAD_PG_REF: dict = {}


def _reserve_head_placement_group(server_name: str, tp_size: int, pp_size: int, pack_strategy: str):
    """Pre-reserve DP rank 0's placement group pinned to this node."""
    import ray
    from ray._private.services import get_node_ip_address

    node_ip = get_node_ip_address()
    node_resource_key = f"node:{node_ip}"

    placement_strategy = "STRICT_PACK" if pack_strategy in ("strict", "fill") else "PACK"
    world_size = tp_size * pp_size

    # Each bundle asks for 1 GPU + a small share of this node's node:<ip>
    # auto-resource, which pins the bundle to this specific node. This
    # guarantees the api_server (which we launch in this process) is
    # colocated with DP rank 0's compute.
    bundles = [{"GPU": 1.0, node_resource_key: 0.01} for _ in range(world_size)] + [
        {"CPU": 1.0, node_resource_key: 0.01}
    ]
    pg = ray.util.placement_group(
        name=f"{server_name}_dp_rank_0",
        strategy=placement_strategy,
        bundles=bundles,
    )
    ray.get(pg.ready())

    return pg, node_ip


def _patch_signal_for_thread_safety() -> None:
    """No-op signal installation.

    vLLM's ``launcher.serve_http`` installs SIGINT/SIGTERM handlers via
    ``loop.add_signal_handler`` + ``signal.signal``. Neither is safe when
    a nested event loop runs off the main thread. We run on main thread
    here, so this is defensive — only needed if a future caller wraps
    this entrypoint in a background thread.
    """
    signal.signal = lambda *a, **kw: None


def _build_vllm_argv(args: argparse.Namespace, extra: Sequence[str]) -> list[str]:
    """Build the argv that vLLM's ``make_arg_parser()`` will parse."""
    argv = [
        f"--model={args.model}",
        f"--served-model-name={args.model}",
        "--trust-remote-code",
        "--host=0.0.0.0",
        f"--port={args.port}",
        f"--tensor-parallel-size={args.num_gpus * args.num_nodes}",
    ]
    if args.no_verbose:
        argv.extend(["--disable-log-requests", "--disable-log-stats"])
    argv.extend(extra)
    return argv


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve vLLM model with DP-on-Ray placement-group patch")
    parser.add_argument("--model", help="Path to the model or a HF model name")
    parser.add_argument("--num_gpus", type=int, required=True)
    parser.add_argument("--num_nodes", type=int, required=False, default=1)
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--no_verbose", action="store_true")
    parser.add_argument(
        "--server_name",
        type=str,
        default="nemo_skills_vllm",
        help="Prefix for placement group names (for debugging in Ray dashboard)",
    )
    args, unknown = parser.parse_known_args()

    print(f"Deploying model {args.model} via serve_vllm_dp_ray")
    print("Starting OpenAI server (in-process, with DP-on-Ray PG patch)")

    # ------------------------------------------------------------------------------
    # Parse vLLM args first so we know TP / PP / DP sizing before Ray init.
    # ------------------------------------------------------------------------------
    from vllm.entrypoints.openai.api_server import (
        FlexibleArgumentParser,
        cli_env_setup,
        make_arg_parser,
        validate_parsed_serve_args,
    )

    cli_env_setup()
    vllm_parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    vllm_parser = make_arg_parser(vllm_parser)
    vllm_args = vllm_parser.parse_args(_build_vllm_argv(args, unknown))
    validate_parsed_serve_args(vllm_args)

    tp_size = getattr(vllm_args, "tensor_parallel_size", 1)
    pp_size = getattr(vllm_args, "pipeline_parallel_size", 1)
    dp_size = getattr(vllm_args, "data_parallel_size", 1)

    # Default pack strategy — matches vLLM's Ray DP path. Callers can
    # override via VLLM_RAY_DP_PACK_STRATEGY in the submitting env.
    pack_strategy = os.environ.setdefault("VLLM_RAY_DP_PACK_STRATEGY", "strict")

    print(f"[serve_vllm_dp_ray] tp={tp_size} pp={pp_size} dp={dp_size} pack_strategy={pack_strategy}")

    # ------------------------------------------------------------------------------
    # Attach to the existing Ray cluster (started by get_ray_server_cmd for
    # multi-node deployments; auto-started if single-node).
    # ------------------------------------------------------------------------------
    import ray

    ray.init(address="auto" if args.num_nodes > 1 else None, ignore_reinit_error=True)

    # ------------------------------------------------------------------------------
    # Reserve DP rank 0's placement group on THIS node and set
    # VLLM_DP_MASTER_IP so DP engines find the coordinator.
    # ------------------------------------------------------------------------------
    head_pg, node_ip = _reserve_head_placement_group(
        server_name=args.server_name,
        tp_size=tp_size,
        pp_size=pp_size,
        pack_strategy=pack_strategy,
    )
    os.environ["VLLM_DP_MASTER_IP"] = node_ip
    print(f"[serve_vllm_dp_ray] Reserved head PG on {node_ip} (DP rank 0)")

    _HEAD_PG_REF["pg"] = head_pg
    _HEAD_PG_REF["server_name"] = args.server_name

    # ------------------------------------------------------------------------------
    # Apply monkey-patches, then launch vLLM's OpenAI api_server in-process.
    # ------------------------------------------------------------------------------
    _apply_vllm_patches()
    _patch_signal_for_thread_safety()

    from vllm.entrypoints.openai.api_server import run_server

    # Optional pipe-through for log filtering — default path leaves
    # stdout untouched; callers who want the " | grep -v '200 OK' "
    # behaviour from serve_vllm.py should use --no_verbose + vLLM's
    # --disable-log-requests instead (cleaner than a shell pipe here).

    # Keep argv printable for debugging.
    print(f"[serve_vllm_dp_ray] vLLM argv: {shlex.join(sys.argv[1:])}")

    asyncio.run(run_server(vllm_args))


if __name__ == "__main__":
    main()
