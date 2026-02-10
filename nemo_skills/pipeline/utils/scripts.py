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

"""
Script classes for NeMo-Skills pipeline components.

These classes wrap NeMo-Run's run.Script interface to provide typed, reusable
job components (servers, clients, sandboxes) with explicit fields and
cross-component reference support for heterogeneous jobs.

Example:
    # Create a server script with automatic port allocation
    server = ServerScript(
        server_type="vllm",
        model_path="/models/llama-8b",
        cluster_config=cluster_config,
        num_gpus=8,
    )

    # Create a client that references the server
    client = GenerationClientScript(
        output_dir="/results",
        input_file="/data/input.jsonl",
        server=server,  # Cross-component reference
    )

    # Use in Command objects
    Command(script=server, container="vllm", ...)
    Command(script=client, container="nemo-skills", ...)
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import nemo_run as run

from nemo_skills.pipeline.utils.commands import sandbox_command
from nemo_skills.pipeline.utils.exp import install_packages_wrap
from nemo_skills.pipeline.utils.generation import get_generation_cmd
from nemo_skills.pipeline.utils.server import get_free_port, get_server_command
from nemo_skills.utils import get_logger_name

if TYPE_CHECKING:
    # Avoid circular imports for type hints
    pass

LOG = logging.getLogger(get_logger_name(__file__))


@dataclass
class BaseJobScript(run.Script):
    """Base class for job component scripts with heterogeneous job support.

    This class provides:
    - het_group_index tracking for cross-component references in heterogeneous SLURM jobs
    - hostname_ref() method for getting hostnames in het jobs
    - Common pattern for Script initialization

    Attributes:
        het_group_index: Index in heterogeneous job group (set by Pipeline at runtime)
        span_group_nodes: Whether to span all nodes from the group's HardwareConfig.
            When False (default), the script runs on 1 node regardless of group config.
            When True, the script spans all nodes specified in the group's num_nodes.
            This is important for multi-node setups with --overlap where the server
            needs multiple nodes but client/sandbox should run on the master node only.
        num_tasks_override: Override the group's num_tasks for this specific script.
            When set, this script's srun will use this value for --ntasks-per-node
            instead of the group's HardwareConfig.num_tasks. Useful when multiple
            scripts in a CommandGroup need different task configurations (e.g.,
            vLLM servers needing 2 tasks per node while Gym client needs 1).
    """

    het_group_index: Optional[int] = field(default=None, init=False, repr=False)
    span_group_nodes: bool = False  # Default: run on 1 node
    num_tasks_override: Optional[int] = None  # Per-script task count override
    installation_command: Optional[str] = None
    entrypoint: str = field(default="bash", init=False)

    def __post_init__(self):
        """Wrap inline command with installation_command if provided."""
        if not self.installation_command:
            return

        if callable(self.inline):
            original_inline = self.inline

            def wrapped_inline():
                result = original_inline()
                if isinstance(result, tuple):
                    command, metadata = result
                    return install_packages_wrap(command, self.installation_command), metadata
                return install_packages_wrap(result, self.installation_command)

            self.set_inline(wrapped_inline)
        elif isinstance(self.inline, str):
            self.set_inline(install_packages_wrap(self.inline, self.installation_command))

    def set_inline(self, command: Union[str, Callable, run.Script]) -> None:
        """Set the inline command safely on frozen dataclass."""
        object.__setattr__(self, "inline", command)

    def hostname_ref(self) -> str:
        """Get hostname reference for hetjob cross-component communication.

        Returns a shell variable reference that resolves to the master node hostname
        for this het group. Uses environment variables automatically exported by nemo-run:
            SLURM_MASTER_NODE_HET_GROUP_0, SLURM_MASTER_NODE_HET_GROUP_1, etc.

        These are set via:
            export SLURM_MASTER_NODE_HET_GROUP_N=$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_N | head -n1)
        """
        if self.het_group_index is None:
            return "127.0.0.1"  # Local fallback for non-heterogeneous jobs

        # Use the environment variable exported by nemo-run
        return f"${{SLURM_MASTER_NODE_HET_GROUP_{self.het_group_index}:-localhost}}"


@dataclass(kw_only=True)
class ServerScript(BaseJobScript):
    """Script for model inference servers (vLLM, TRT-LLM, SGLang, etc.).

    This script wraps server command builders and provides:
    - Automatic port allocation if not specified
    - Type-safe server configuration
    - Cross-component address sharing (get_address())
    - Resource requirement tracking (num_gpus, num_nodes, num_tasks)

    Attributes:
        server_type: Type of server (vllm, trtllm, sglang, megatron, openai, etc.)
        model_path: Path to model weights or model name for API services
        cluster_config: Cluster configuration dictionary
        num_gpus: Number of GPUs required (default: 8)
        num_nodes: Number of nodes required (default: 1)
        server_args: Additional server-specific arguments
        server_entrypoint: Custom server entrypoint script (optional)
        port: Server port (allocated automatically if None)
        allocate_port: Whether to allocate port automatically (default: True)
        num_tasks: Number of MPI tasks (computed in __post_init__)
        log_prefix: Prefix for log files (default: "server")

    Example:
        # Basic usage
        server = ServerScript(
            server_type="vllm",
            model_path="/models/llama-3-8b",
            cluster_config=cluster_config,
            num_gpus=8,
        )

        # Access allocated port
        print(f"Server will run on port {server.port}")

        # Get full address for client connection
        address = server.get_address()  # Returns "hostname:port"
    """

    server_type: str
    model_path: str
    cluster_config: Dict
    num_gpus: int = 8
    num_nodes: int = 1
    server_args: str = ""
    server_entrypoint: Optional[str] = None  # Custom server entrypoint script
    port: Optional[int] = None
    allocate_port: bool = True

    # Server spans all group nodes (e.g., for distributed inference)
    span_group_nodes: bool = True

    # Computed fields (set in __post_init__)
    num_tasks: int = field(init=False, repr=False)
    log_prefix: str = field(default="server", init=False)

    def __post_init__(self):
        """Initialize server script.

        - Allocates port if not provided
        - Builds server command using get_server_command()
        - Sets self.inline to the command string
        - Computes num_tasks from server command builder
        """
        # Allocate port if not provided
        if self.port is None and self.allocate_port:
            self.port = get_free_port(strategy="random")
            LOG.debug(f"Allocated port {self.port} for {self.server_type} server")

        # Build server command
        cmd, self.num_tasks = get_server_command(
            server_type=self.server_type,
            num_gpus=self.num_gpus,
            num_nodes=self.num_nodes,
            model_path=self.model_path,
            cluster_config=self.cluster_config,
            server_port=self.port,
            server_args=self.server_args,
            server_entrypoint=self.server_entrypoint,
        )

        self.set_inline(cmd)
        super().__post_init__()

    def get_address(self) -> str:
        """Get server address for client connections.

        Returns hostname:port string that clients can use to connect.
        In heterogeneous jobs, hostname_ref() returns a bash expression
        that resolves at runtime.

        Returns:
            Server address in format "hostname:port"

        Example:
            # Use in client command
            client_cmd = f"python client.py --server-url http://{server.get_address()}"
        """
        return f"{self.hostname_ref()}:{self.port}"


@dataclass(kw_only=True)
class SandboxScript(BaseJobScript):
    """Script for code execution sandbox container.

    The sandbox provides a secure environment for executing LLM-generated code.
    This script wraps sandbox command builders and provides:
    - Automatic port allocation
    - Mount configuration (can optionally keep mounts, though risky)
    - Type-safe sandbox configuration

    Attributes:
        cluster_config: Cluster configuration dictionary
        port: Sandbox port (allocated automatically if None)
        keep_mounts: Whether to keep filesystem mounts (default: False, risky if True).
                     Note: This is stored for documentation but actually handled at
                     the executor level, not in the sandbox command itself.
        allocate_port: Whether to allocate port automatically (default: True)
        log_prefix: Prefix for log files (default: "sandbox")

    Example:
        sandbox = SandboxScript(
            cluster_config=cluster_config,
            keep_mounts=False,  # Safer: sandbox has no access to mounted paths
        )

        # Client can reference sandbox port
        client = GenerationClientScript(..., sandbox=sandbox)
    """

    cluster_config: Dict
    port: Optional[int] = None
    keep_mounts: bool = False
    allocate_port: bool = True
    env_overrides: Optional[List[str]] = None  # Extra env vars in KEY=VALUE form

    # Sandbox spans all group nodes (e.g., for multi-node generate jobs)
    span_group_nodes: bool = True

    log_prefix: str = field(default="sandbox", init=False)

    def __post_init__(self):
        """Initialize sandbox script.

        - Allocates port if not provided
        - Builds sandbox command using sandbox_command()
        - Sets self.inline to a callable that returns command and environment vars
        """
        # Allocate port if not provided
        if self.port is None and self.allocate_port:
            self.port = get_free_port(strategy="random")
            LOG.debug(f"Allocated port {self.port} for sandbox")

        # Build sandbox command and metadata (including environment vars)
        # Note: keep_mounts is handled at the executor level, not in the command itself
        cmd, metadata = sandbox_command(
            cluster_config=self.cluster_config,
            port=self.port,
        )

        # Use a callable to return both command and environment variables
        # This ensures the sandbox's LISTEN_PORT and NGINX_PORT are properly set
        def build_cmd() -> Tuple[str, Dict]:
            env = dict(metadata.get("environment", {}))
            # Apply user-specified environment overrides
            if self.env_overrides:
                for override in self.env_overrides:
                    key, value = override.split("=", 1)
                    env[key] = value
            return cmd, {"environment": env}

        self.set_inline(build_cmd)
        super().__post_init__()


@dataclass(kw_only=True)
class GenerationClientScript(BaseJobScript):
    """Script for LLM generation/inference client.

    This script wraps generation command builders and provides:
    - Cross-component references to multiple servers and sandbox
    - Lazy command building for runtime hostname resolution
    - Type-safe generation configuration
    - Environment variable handling for sandbox/server communication

    Attributes:
        output_dir: Directory for output files
        input_file: Input JSONL file (mutually exclusive with input_dir)
        input_dir: Input directory (mutually exclusive with input_file)
        extra_arguments: Additional arguments for generation script
        random_seed: Random seed for sampling (optional)
        chunk_id: Chunk ID for parallel processing (optional)
        num_chunks: Total number of chunks (required if chunk_id set)
        preprocess_cmd: Command to run before generation (optional)
        postprocess_cmd: Command to run after generation (optional)
        wandb_parameters: WandB logging configuration (optional)
        with_sandbox: Whether sandbox is enabled
        script: Module or file path for generation script (default: nemo_skills.inference.generate)
        servers: List of ServerScript references (None for pre-hosted servers)
        server_addresses_prehosted: Addresses for pre-hosted servers (parallel to servers list)
        model_names: Model names for multi-model generation (optional)
        server_types: Server types for multi-model generation (optional)
        sandbox: Reference to SandboxScript for cross-component communication (optional)
        log_prefix: Prefix for log files (default: "main")

    Examples:
        # Single server
        client = GenerationClientScript(
            output_dir="/results",
            input_file="/data/input.jsonl",
            servers=[server_script],
            model_names=["llama-8b"],
            server_types=["vllm"],
        )

        # Multi-model with self-hosted and pre-hosted servers
        client = GenerationClientScript(
            output_dir="/results",
            input_file="/data/input.jsonl",
            servers=[server1, server2, None],  # None = pre-hosted
            server_addresses_prehosted=["", "", "https://api.openai.com"],
            model_names=["llama-8b", "llama-70b", "gpt-4"],
            server_types=["vllm", "vllm", "openai"],
            sandbox=sandbox_script,
            with_sandbox=True,
        )
    """

    output_dir: str
    input_file: Optional[str] = None
    input_dir: Optional[str] = None
    extra_arguments: str = ""
    random_seed: Optional[int] = None
    chunk_id: Optional[int] = None
    num_chunks: Optional[int] = None
    preprocess_cmd: Optional[str] = None
    postprocess_cmd: Optional[str] = None
    wandb_parameters: Optional[Dict] = None
    with_sandbox: bool = False
    script: str = "nemo_skills.inference.generate"
    requirements: Optional[list[str]] = None

    # Cross-component references for single/multi-model
    servers: Optional[List[Optional["ServerScript"]]] = None
    server_addresses_prehosted: Optional[List[str]] = None
    model_names: Optional[List[str]] = None
    server_types: Optional[List[str]] = None
    sandbox: Optional["SandboxScript"] = None

    log_prefix: str = field(default="main", init=False)

    def __post_init__(self):
        """Initialize generation client script with lazy command building.

        Builds command lazily via a callable that is evaluated when het_group_index
        is assigned, allowing hostname_ref() to resolve correctly for heterogeneous jobs.

        This works for both cases:
        - With cross-refs: Resolves server hostnames and sandbox ports at runtime
        - Without cross-refs: Just builds the command string (no runtime resolution needed)
        """

        def build_cmd() -> Tuple[str, Dict]:
            """Build command at runtime when cross-refs are resolved."""
            env_vars = {}

            # Add sandbox port to environment if sandbox is referenced
            if self.sandbox:
                env_vars["NEMO_SKILLS_SANDBOX_PORT"] = str(self.sandbox.port)

            # Build server addresses if servers are provided
            server_addresses = None
            if self.servers is not None:
                server_addresses = []
                for server_idx, server_script in enumerate(self.servers):
                    if server_script is not None:
                        # Self-hosted: construct address from hostname and port refs
                        addr = f"{server_script.hostname_ref()}:{server_script.port}"
                    else:
                        # Pre-hosted: use the address from server_addresses_prehosted
                        addr = self.server_addresses_prehosted[server_idx]
                    server_addresses.append(addr)

            # Build generation command
            cmd = get_generation_cmd(
                output_dir=self.output_dir,
                input_file=self.input_file,
                input_dir=self.input_dir,
                extra_arguments=self.extra_arguments,
                random_seed=self.random_seed,
                chunk_id=self.chunk_id,
                num_chunks=self.num_chunks,
                preprocess_cmd=self.preprocess_cmd,
                postprocess_cmd=self.postprocess_cmd,
                wandb_parameters=self.wandb_parameters,
                with_sandbox=self.with_sandbox,
                script=self.script,
                requirements=self.requirements,
                # Multi-model parameters (None for single-model)
                server_addresses=server_addresses,
                model_names=self.model_names,
                server_types=self.server_types,
            )

            # Return command and runtime metadata (environment vars)
            return cmd, {"environment": env_vars}

        # Always use lazy command building
        self.set_inline(build_cmd)
        super().__post_init__()


@dataclass(kw_only=True)
class NemoGymRolloutsScript(BaseJobScript):
    """Script for running NeMo Gym rollout collection.

    This script orchestrates the full rollout collection workflow:
    1. Starts ng_run in background to spin up NeMo Gym servers
    2. Polls ng_status until all servers are healthy
    3. Runs ng_collect_rollouts to collect rollouts
    4. Keeps ng_run running (cleanup handled externally)

    Attributes:
        config_paths: List of YAML config file paths for ng_run
        input_file: Input JSONL file path for rollout collection
        output_file: Output JSONL file path for rollouts
        extra_arguments: Additional Hydra overrides passed to both ng_run and ng_collect_rollouts
        server: Optional ServerScript reference for policy model server
        server_address: Optional pre-hosted server address
        sandbox: Optional SandboxScript reference for sandbox port
        log_prefix: Prefix for log files (default: "nemo_gym")

    Example:
        script = NemoGymRolloutsScript(
            config_paths=[
                "resources_servers/ns_tools/configs/ns_tools.yaml",
                "resources_servers/math_with_judge/configs/math_with_judge.yaml",
            ],
            input_file="/data/input.jsonl",
            output_file="/data/rollouts.jsonl",
            extra_arguments="+agent_name=ns_tools_simple_agent +limit=10",
            server=server_script,
            sandbox=sandbox_script,
        )
    """

    config_paths: List[str]
    input_file: str
    output_file: str
    extra_arguments: str = ""
    server: Optional["ServerScript"] = None
    server_address: Optional[str] = None
    sandbox: Optional["SandboxScript"] = None
    gym_path: str = "/opt/NeMo-RL/3rdparty/Gym-workspace/Gym"
    policy_api_key: str = "dummy"  # API key for policy server (can be dummy for local)
    policy_model_name: Optional[str] = None  # Model name override for policy server

    log_prefix: str = field(default="nemo_gym", init=False)

    def __post_init__(self):
        """Initialize the combined ng_run + ng_collect_rollouts script."""

        def build_cmd() -> Tuple[str, Dict]:
            """Build the full rollout collection command."""
            # Build config_paths argument
            config_paths_str = ",".join(self.config_paths)

            # Build ng_run command parts
            ng_run_parts = [
                "ng_run",
                f'"+config_paths=[{config_paths_str}]"',
            ]

            # Add policy server URL if we have a server reference or address
            if self.server is not None:
                server_addr = f"http://{self.server.hostname_ref()}:{self.server.port}/v1"
                ng_run_parts.append(f'+policy_base_url="{server_addr}"')
            elif self.server_address is not None:
                ng_run_parts.append(f'+policy_base_url="{self.server_address}"')

            # Add policy API key (required by some configs)
            ng_run_parts.append(f'+policy_api_key="{self.policy_api_key}"')

            # Add policy model name (required by configs)
            if self.policy_model_name:
                ng_run_parts.append(f'+policy_model_name="{self.policy_model_name}"')

            # Add extra arguments to ng_run
            if self.extra_arguments:
                ng_run_parts.append(self.extra_arguments)

            ng_run_cmd = " ".join(ng_run_parts)

            # Build ng_collect_rollouts command
            ng_collect_parts = [
                "ng_collect_rollouts",
                f'+input_jsonl_fpath="{self.input_file}"',
                f'+output_jsonl_fpath="{self.output_file}"',
            ]

            # Add extra arguments to ng_collect_rollouts
            if self.extra_arguments:
                ng_collect_parts.append(self.extra_arguments)

            ng_collect_cmd = " ".join(ng_collect_parts)

            # Compute the vLLM server URL for the wait check
            if self.server is not None:
                vllm_server_url = f"http://{self.server.hostname_ref()}:{self.server.port}/v1"
            elif self.server_address is not None:
                vllm_server_url = self.server_address
            else:
                vllm_server_url = ""

            # Build the full bash script that:
            # 1. Installs NeMo Gym from 3rdparty/Gym-workspace/Gym
            # 2. Waits for vLLM server to be ready
            # 3. Starts ng_run in background
            # 4. Polls ng_status until healthy (with early failure detection)
            # 5. Runs ng_collect_rollouts
            cmd = f"""set -e
set -o pipefail

echo "=== Installing NeMo Gym ==="
cd {self.gym_path} || {{ echo "ERROR: Failed to cd to Gym directory"; exit 1; }}
uv venv --python 3.12 --allow-existing .venv || {{ echo "ERROR: Failed to create venv"; exit 1; }}
source .venv/bin/activate || {{ echo "ERROR: Failed to activate venv"; exit 1; }}
uv sync --active --extra dev || {{ echo "ERROR: Failed to sync dependencies"; exit 1; }}
echo "NeMo Gym installed successfully"

# Disable pipefail for the polling loop (grep may return non-zero)
set +o pipefail

# Wait for vLLM server to be ready before starting ng_run
# Note: --kill-on-bad-exit in srun ensures job fails if vLLM crashes
VLLM_SERVER_URL="{vllm_server_url}"
if [ -n "$VLLM_SERVER_URL" ]; then
    echo "=== Waiting for vLLM server at $VLLM_SERVER_URL ==="
    while [ $(curl -s -o /dev/null -w "%{{http_code}}" "$VLLM_SERVER_URL/models" 2>/dev/null) -ne 200 ]; do
        sleep 10
    done
    echo "vLLM server is ready!"
fi

echo "=== Starting NeMo Gym servers ==="
{ng_run_cmd} &
NG_RUN_PID=$!
echo "ng_run PID: $NG_RUN_PID"

echo "Waiting for NeMo Gym servers..."
LAST_STATUS=""
while true; do
    # Check if ng_run process died - let the failure cascade naturally
    if ! kill -0 $NG_RUN_PID 2>/dev/null; then
        echo "ERROR: ng_run process exited unexpectedly"
        wait $NG_RUN_PID 2>/dev/null  # Get exit code
        exit 1
    fi

    STATUS_OUTPUT=$(ng_status 2>&1)

    if echo "$STATUS_OUTPUT" | grep -q "healthy, 0 unhealthy"; then
        echo "All servers ready!"
        break
    fi

    # Only print status when it changes (reduce verbosity)
    CURRENT_STATUS=$(echo "$STATUS_OUTPUT" | grep -oE '[0-9]+ healthy' | head -1 || echo "starting")
    if [ "$CURRENT_STATUS" != "$LAST_STATUS" ]; then
        echo "Server status: $CURRENT_STATUS"
        LAST_STATUS="$CURRENT_STATUS"
    fi

    sleep 10
done

echo "=== Running rollout collection ==="
echo "Input file: {self.input_file}"
echo "Output file: {self.output_file}"
mkdir -p "$(dirname "{self.output_file}")"
echo "Output directory created: $(dirname "{self.output_file}")"
echo "Running: {ng_collect_cmd}"
{ng_collect_cmd} || {{ echo "ERROR: ng_collect_rollouts failed"; kill $NG_RUN_PID 2>/dev/null || true; exit 1; }}

echo "=== Rollout collection complete ==="
echo "Output: {self.output_file}"

echo "=== Cleaning up ==="
kill $NG_RUN_PID 2>/dev/null || true
echo "Servers terminated."
"""
            # Build environment variables for sandbox connection
            # YAML configs use ${oc.env:NEMO_SKILLS_SANDBOX_HOST,127.0.0.1} and
            # ${oc.env:NEMO_SKILLS_SANDBOX_PORT,6000} to resolve these
            env_vars = {}
            if self.sandbox is not None:
                env_vars["NEMO_SKILLS_SANDBOX_HOST"] = self.sandbox.hostname_ref()
                env_vars["NEMO_SKILLS_SANDBOX_PORT"] = str(self.sandbox.port)

            return cmd.strip(), {"environment": env_vars}

        self.set_inline(build_cmd)
        super().__post_init__()


@dataclass(kw_only=True)
class MultiVLLMServerScript(BaseJobScript):
    """Script for deploying multiple independent vLLM servers for Gym routing.

    This script enables data-parallel vLLM deployments where Gym routes requests
    across multiple independent vLLM server replicas. Each replica runs on its own
    set of GPUs and handles requests independently.

    Supports configurations like:
    - 1 server per node with TP=8 (full node per server)
    - 2 servers per node with TP=4 each (2 servers sharing a node)
    - 4 servers per node with TP=2 each
    - 8 servers per node with TP=1 each

    The script uses SLURM environment variables to determine which server instance
    to start on each task:
    - SLURM_PROCID: Global task ID (0 to num_nodes * servers_per_node - 1)
    - SLURM_LOCALID: Task ID within node (0 to servers_per_node - 1)
    - SLURM_NODEID: Node ID (0 to num_nodes - 1)

    Attributes:
        model_path: Path to model weights or HuggingFace model name
        cluster_config: Cluster configuration dictionary
        num_nodes: Number of nodes to use
        servers_per_node: Number of vLLM servers per node
        gpus_per_server: GPUs per server (tensor_parallel_size)
        server_args: Additional vLLM server arguments
        base_port: Starting port (servers use base_port + global_rank)

    Example:
        # 4 nodes × 2 servers per node = 8 total replicas, each with TP=4
        servers = MultiVLLMServerScript(
            model_path="/models/llama-70b",
            cluster_config=cluster_config,
            num_nodes=4,
            servers_per_node=2,
            gpus_per_server=4,
            server_args="--dtype auto --max-model-len 8192",
        )

        # Use in CommandGroup
        group = CommandGroup(
            commands=[
                Command(script=servers, container="vllm", name="vllm"),
                Command(script=gym_client, container="nemo-skills", name="gym"),
            ],
            hardware=HardwareConfig(
                num_nodes=4,
                num_gpus=8,
                num_tasks=2,  # servers_per_node
            ),
        )
    """

    model_path: str
    cluster_config: Dict
    num_nodes: int = 1
    servers_per_node: int = 1
    gpus_per_server: int = 8
    server_args: str = ""
    base_port: Optional[int] = None
    # Mitigations for vLLM v1 shared-memory communicator instability on some clusters.
    # When enabled, we export env vars that (if supported by the vLLM version) disable
    # SHM-based broadcast paths, which are sensitive to /dev/shm sizing and IPC quirks.
    disable_shm_broadcast: bool = False
    # Print /dev/shm diagnostics at startup to help debug node-to-node differences.
    print_shm_diagnostics: bool = False

    # Spans all nodes - each SLURM task runs one server
    span_group_nodes: bool = True

    # Internal tracking
    _total_replicas: int = field(init=False, repr=False, default=0)
    _ports: List[int] = field(init=False, repr=False, default_factory=list)
    log_prefix: str = field(default="vllm_servers", init=False)

    def __post_init__(self):
        """Build command for distributed vLLM servers."""
        self._total_replicas = self.num_nodes * self.servers_per_node

        # Set num_tasks_override to ensure this script gets the right task count
        self.num_tasks_override = self.servers_per_node

        # Allocate ports
        if self.base_port is None:
            self.base_port = get_free_port(strategy="random")
        self._ports = [self.base_port + i for i in range(self._total_replicas)]

        ports_str = " ".join(str(p) for p in self._ports)

        # Build command that runs one server per SLURM task
        disable_shm_broadcast = "true" if self.disable_shm_broadcast else "false"
        print_shm_diagnostics = "true" if self.print_shm_diagnostics else "false"
        cmd = f'''
#!/bin/bash
set -e

# Configuration
PORTS=({ports_str})
GPUS_PER_SERVER={self.gpus_per_server}
SERVERS_PER_NODE={self.servers_per_node}
DISABLE_SHM_BROADCAST={disable_shm_broadcast}
PRINT_SHM_DIAGNOSTICS={print_shm_diagnostics}

# SLURM environment
# SLURM_PROCID: Global task ID (0 to num_nodes * servers_per_node - 1)
# SLURM_LOCALID: Task ID within this node (0 to servers_per_node - 1)
# SLURM_NODEID: Node ID (0 to num_nodes - 1)
GLOBAL_RANK=${{SLURM_PROCID:-0}}
LOCAL_RANK=${{SLURM_LOCALID:-0}}
NODE_ID=${{SLURM_NODEID:-0}}

# Calculate which GPUs this server should use
# E.g., with 2 servers per node and 4 GPUs each:
#   Local rank 0 → GPUs 0,1,2,3
#   Local rank 1 → GPUs 4,5,6,7
GPU_START=$((LOCAL_RANK * GPUS_PER_SERVER))
GPU_END=$((GPU_START + GPUS_PER_SERVER - 1))
GPU_LIST=$(seq -s, $GPU_START $GPU_END)

# Get port for this server
MY_PORT=${{PORTS[$GLOBAL_RANK]}}

echo "=== vLLM Server Configuration ==="
echo "Global Rank: $GLOBAL_RANK"
echo "Node ID: $NODE_ID"
echo "Local Rank: $LOCAL_RANK"
echo "Port: $MY_PORT"
echo "GPUs: $GPU_LIST (CUDA_VISIBLE_DEVICES)"
echo "Tensor Parallel Size: {self.gpus_per_server}"
echo "Model: {self.model_path}"
echo "================================="

# Debug /dev/shm and IPC-related info (helps with node-to-node variance)
if [ "$PRINT_SHM_DIAGNOSTICS" = "true" ]; then
  echo ""
  echo "=== /dev/shm diagnostics (before vLLM start) ==="
  df -h /dev/shm || true
  ls -ld /dev/shm || true
  mount | grep -E " /dev/shm " || true
  echo "ulimit -n: $(ulimit -n || true)"
  echo "==============================================="
  echo ""
fi

# Set GPU visibility
export CUDA_VISIBLE_DEVICES=$GPU_LIST

# Mitigation: attempt to disable vLLM SHM broadcast transport (best-effort, version-dependent).
# These env vars are intentionally safe: if vLLM doesn't recognize them, they are ignored.
if [ "$DISABLE_SHM_BROADCAST" = "true" ]; then
  export VLLM_DISABLE_SHM_BROADCAST=1
  export VLLM_USE_SHM_BROADCAST=0
fi

# Start vLLM server
python3 -m vllm.entrypoints.openai.api_server \\
    --model "{self.model_path}" \\
    --host "0.0.0.0" \\
    --port "$MY_PORT" \\
    --tensor-parallel-size {self.gpus_per_server} \\
    --trust-remote-code \\
    {self.server_args}
'''

        self.set_inline(cmd)
        super().__post_init__()

    @property
    def total_replicas(self) -> int:
        """Total number of vLLM server replicas."""
        return self._total_replicas

    @property
    def ports(self) -> List[int]:
        """List of ports for all server replicas."""
        return self._ports


@dataclass(kw_only=True)
class GymClientScript(BaseJobScript):
    """Script that starts Gym with routing to multiple vLLM servers.

    This script runs on a single node (the master node) and:
    1. Waits for all vLLM servers to be healthy
    2. Builds the list of server URLs
    3. Exports GYM_VLLM_BASE_URLS environment variable
    4. Starts Gym with the configured command

    Attributes:
        servers: Reference to MultiVLLMServerScript for server URLs
        gym_command: Command to run Gym (e.g., "python -m nemo_gym.server ...")
        health_check_interval: Seconds between health check attempts (default: 10)

    Example:
        gym_client = GymClientScript(
            servers=vllm_servers,
            gym_command="python -m nemo_gym.server --config /path/to/config.yaml",
        )

        # In same CommandGroup as vllm_servers
        group = CommandGroup(
            commands=[
                Command(script=vllm_servers, container="vllm", name="vllm"),
                Command(script=gym_client, container="nemo-skills", name="gym"),
            ],
            ...
        )
    """

    servers: "MultiVLLMServerScript"
    gym_command: str
    health_check_interval: int = 10  # Seconds between health check attempts

    # Runs only on master node, single task
    span_group_nodes: bool = False
    num_tasks_override: int = 1

    log_prefix: str = field(default="gym", init=False)

    def __post_init__(self):
        """Build command that waits for servers then starts Gym."""

        def build_cmd() -> Tuple[str, Dict]:
            ports_str = " ".join(str(p) for p in self.servers.ports)
            num_servers = self.servers.total_replicas
            num_nodes = self.servers.num_nodes
            servers_per_node = self.servers.servers_per_node

            cmd = f"""
#!/bin/bash
set -e

echo "=== Gym Client Starting ==="
echo "Waiting for {num_servers} vLLM servers across {num_nodes} nodes..."
echo "No timeout - will wait until servers are ready or job is cancelled."
echo "If vLLM servers crash, the job will fail."

# Configuration
PORTS=({ports_str})
NUM_NODES={num_nodes}
SERVERS_PER_NODE={servers_per_node}
HEALTH_CHECK_INTERVAL={self.health_check_interval}

# Expand SLURM_JOB_NODELIST without scontrol (not available in containers)
# Uses Python to parse the compressed node list format (e.g., "node[001-004]")
expand_nodelist() {{
    python3 -c "
import re
import sys

nodelist = sys.argv[1]
nodes = []

# Handle comma-separated parts
for part in re.split(r',(?![^\\[]*\\])', nodelist):
    # Check if it has a range like prefix[001-003,005]
    match = re.match(r'(.*)\\[([^\\]]+)\\](.*)', part)
    if match:
        prefix, ranges, suffix = match.groups()
        for r in ranges.split(','):
            if '-' in r:
                start, end = r.split('-')
                width = len(start)
                for i in range(int(start), int(end) + 1):
                    nodes.append(f'{{prefix}}{{i:0{{width}}d}}{{suffix}}')
            else:
                nodes.append(f'{{prefix}}{{r}}{{suffix}}')
    else:
        nodes.append(part)

print(' '.join(nodes))
" "$1"
}}

# Get all node hostnames.
# IMPORTANT: depending on how the step is launched, some clusters populate only one of these.
# We'll pick the candidate that expands to the most nodes, and verify it matches NUM_NODES.
BEST_NODES_STR=""
BEST_COUNT=0
for VAR_NAME in SLURM_JOB_NODELIST SLURM_NODELIST SLURM_STEP_NODELIST; do
    NL="${{!VAR_NAME:-}}"
    if [ -z "$NL" ]; then
        continue
    fi
    EXPANDED="$(expand_nodelist "$NL" || true)"
    COUNT="$(echo "$EXPANDED" | wc -w | tr -d ' ')"
    if [ "$COUNT" -gt "$BEST_COUNT" ]; then
        BEST_COUNT="$COUNT"
        BEST_NODES_STR="$EXPANDED"
        BEST_VAR_NAME="$VAR_NAME"
    fi
done

if [ "$BEST_COUNT" -lt "$NUM_NODES" ]; then
    echo "ERROR: Could not discover enough node hostnames for NUM_NODES=$NUM_NODES"
    echo "  Best candidate: ${{BEST_VAR_NAME:-<none>}} -> $BEST_COUNT nodes: '${{BEST_NODES_STR:-}}'"
        echo "  SLURM_JOB_NODELIST='${{SLURM_JOB_NODELIST:-}}'"
        echo "  SLURM_NODELIST='${{SLURM_NODELIST:-}}'"
        echo "  SLURM_STEP_NODELIST='${{SLURM_STEP_NODELIST:-}}'"
    exit 1
fi

read -ra NODES <<< "$BEST_NODES_STR"
echo "Nodes (from $BEST_VAR_NAME): ${{NODES[*]}}"

# Build URL list and wait for each server (no timeout - wait indefinitely)
URLS=""
GLOBAL_IDX=0
START_TIME=$(date +%s)

for NODE_IDX in $(seq 0 $((NUM_NODES - 1))); do
    NODE=${{NODES[$NODE_IDX]}}
    if [ -z "$NODE" ]; then
        echo "ERROR: Empty NODE at index $NODE_IDX. Nodes: ${{NODES[*]}}"
        exit 1
    fi

    for LOCAL_IDX in $(seq 0 $((SERVERS_PER_NODE - 1))); do
        PORT=${{PORTS[$GLOBAL_IDX]}}
        URL="http://${{NODE}}:${{PORT}}/v1"

        echo "Waiting for server $GLOBAL_IDX at $URL..."
        ATTEMPT=0
        while true; do
            ATTEMPT=$((ATTEMPT + 1))
            if curl -s "${{URL}}/health" > /dev/null 2>&1; then
                ELAPSED=$(($(date +%s) - START_TIME))
                echo "  ✓ Server $GLOBAL_IDX is ready (after ${{ELAPSED}}s total)"
                break
            fi
            # Log progress every 60 seconds
            if [ $((ATTEMPT % (60 / HEALTH_CHECK_INTERVAL))) -eq 0 ]; then
                ELAPSED=$(($(date +%s) - START_TIME))
                echo "  ... still waiting for server $GLOBAL_IDX (${{ELAPSED}}s elapsed)"
            fi
            sleep $HEALTH_CHECK_INTERVAL
        done

        if [ -n "$URLS" ]; then URLS="$URLS,"; fi
        URLS="${{URLS}}$URL"
        GLOBAL_IDX=$((GLOBAL_IDX + 1))
    done
done

TOTAL_TIME=$(($(date +%s) - START_TIME))
echo ""
echo "=== All {num_servers} servers ready! (took ${{TOTAL_TIME}}s) ==="
echo "Server URLs: $URLS"
echo ""

# Export for Gym to use
export GYM_VLLM_BASE_URLS="$URLS"

# Run Gym
echo "Starting Gym..."
{self.gym_command}
"""

            return cmd.strip(), {"environment": {}}

        self.set_inline(build_cmd)
        super().__post_init__()
