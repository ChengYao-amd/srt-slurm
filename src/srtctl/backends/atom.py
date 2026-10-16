# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ROCm ATOM inference backend."""

from __future__ import annotations

import builtins
import json
from collections.abc import Mapping, Sequence
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from marshmallow import Schema
from marshmallow_dataclass import dataclass

from srtctl.backends.base import Backend, BoundRolesField, RoleSettings, role_args
from srtctl.ports import DYN_SYSTEM_PORT_BASE, LMCACHE_SERVER_PORT

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import ProfilingConfig
    from srtctl.core.topology import Endpoint, NodePortAllocator, Process

# ATOM arguments srtctl derives from the allocation; a role cannot set them.
MANAGED_ARGS = frozenset({"model", "host", "server-port", "tp", "tensor-parallel-size", "kv-transfer-config"})
# ATOM sizes that multiply with -tp into the GPUs one server occupies.
_WORLD_SIZE_ARGS = frozenset({"pipeline-parallel-size", "prefill-context-parallel-size", "data-parallel-size"})


@dataclass(frozen=True)
class AtomBackend(Backend):
    """Launch ``atom.entrypoints.openai_server`` on ROCm workers."""

    # Engine type discriminator.
    type: Literal["atom"] = "atom"
    # The roles this engine runs (`roles.<role>` of the recipe), bound by SrtConfig and
    # never written on `engine:`. Per-role env and native ATOM CLI args are read from here.
    roles: Mapping[str, RoleSettings] = field(default_factory=dict, metadata={"marshmallow_field": BoundRolesField()})
    # KV-transfer connector between prefill and decode workers.
    connector: Literal["mooncake"] = "mooncake"
    # Mooncake transport; unset lets ATOM choose.
    mooncake_protocol: Literal["rdma", "tcp"] | None = None

    Schema: ClassVar[builtins.type[Schema]] = Schema

    def get_served_model_name(self, default: str) -> str:
        """The name ATOM serves: a role's ``served-model-name``, else its literal ``--model``."""
        for mode in ("prefill", "agg", "decode"):
            args = role_args(self.roles, mode)
            name = args.get("served-model-name") or args.get("served_model_name")
            if name:
                return name
        return default

    def allocate_endpoints(
        self,
        num_prefill: int,
        num_decode: int,
        num_agg: int,
        gpus_per_prefill: int,
        gpus_per_decode: int,
        gpus_per_agg: int,
        gpus_per_node: int,
        available_nodes: Sequence[str],
        spread_workers: bool = False,
    ) -> list[Endpoint]:
        from srtctl.core.topology import allocate_endpoints

        return allocate_endpoints(
            num_prefill=num_prefill,
            num_decode=num_decode,
            num_agg=num_agg,
            gpus_per_prefill=gpus_per_prefill,
            gpus_per_decode=gpus_per_decode,
            gpus_per_agg=gpus_per_agg,
            gpus_per_node=gpus_per_node,
            available_nodes=available_nodes,
            spread_workers=spread_workers,
        )

    def endpoints_to_processes(
        self,
        endpoints: list[Endpoint],
        base_sys_port: int = DYN_SYSTEM_PORT_BASE,
        port_allocator: NodePortAllocator | None = None,
        frontend_type: str = "atomesh",
        dynamo_sidecar: bool = False,
    ) -> list[Process]:
        if dynamo_sidecar:
            raise ValueError("ATOM does not support Dynamo sidecars")
        from srtctl.core.topology import endpoints_to_processes

        return endpoints_to_processes(endpoints, base_sys_port=base_sys_port, port_allocator=port_allocator)

    def _kv_transfer_config(
        self,
        process: Process,
        worker_ip: str,
        extra_connectors: list[dict[str, Any]],
        mooncake_config: dict[str, Any],
    ) -> str | None:
        """Mooncake for P/D workers plus the role's extra connectors; several are wrapped in ``multi``.

        ``mooncake_config`` (the role's ``mooncake-connector-config``) adds ATOM connector
        keys such as ``num_worker_threads`` to the Mooncake entry; srtctl's own keys stay its.
        """
        connectors = []
        for connector in extra_connectors:
            if connector.get("kv_connector") == "lmcache_mp":
                # Dial the node-local lmcache-server service unless the recipe set a port.
                extra = {"lmcache.mp.port": LMCACHE_SERVER_PORT, **connector.get("kv_connector_extra_config", {})}
                connector = {**connector, "kv_connector_extra_config": extra}
            connectors.append(connector)
        if process.endpoint_mode in {"prefill", "decode"}:
            if process.nixl_port is None:
                raise ValueError("ATOM P/D worker is missing its Mooncake handshake port")
            mooncake: dict[str, Any] = {
                "kv_role": "kv_producer" if process.endpoint_mode == "prefill" else "kv_consumer",
                "kv_connector": self.connector,
                "proxy_ip": worker_ip,
                "handshake_port": process.nixl_port,
            }
            if self.mooncake_protocol is not None:
                mooncake["protocol"] = self.mooncake_protocol
            if not isinstance(mooncake_config, dict):
                raise ValueError("mooncake-connector-config must be a mapping of ATOM Mooncake connector keys")
            overlap = sorted(mooncake.keys() & mooncake_config.keys())
            if overlap:
                raise ValueError(f"mooncake-connector-config cannot override srtctl-managed key(s): {overlap}")
            connectors.insert(0, {**mooncake, **mooncake_config})
        elif mooncake_config:
            raise ValueError("mooncake-connector-config applies to prefill and decode roles only")
        if not connectors:
            return None
        payload = connectors[0] if len(connectors) == 1 else {"kv_connector": "multi", "connectors": connectors}
        return json.dumps(payload, separators=(",", ":"))

    def build_worker_command(
        self,
        process: Process,
        endpoint_processes: list[Process],
        runtime: RuntimeContext,
        frontend_type: str = "atomesh",
        nsys_prefix: list[str] | None = None,
        dump_config_path: Path | None = None,
        profiling: ProfilingConfig | None = None,
    ) -> list[str]:
        from srtctl.frontends import get_frontend

        frontend = get_frontend(frontend_type)
        if frontend.required_backend != self.type:
            raise ValueError(f"backend.type: atom requires an ATOM frontend such as atomesh (got {frontend_type!r})")
        if len({item.node for item in endpoint_processes}) != 1:
            raise ValueError("ATOM currently requires each logical endpoint to fit on one Slurm node")

        from srtctl.core.slurm import get_hostname_ip

        worker_ip = get_hostname_ip(process.node, runtime.network_interface)
        config = self.get_config_for_mode(process.endpoint_mode)
        extra_connectors = config.pop("extra-kv-connectors", [])
        mooncake_config = config.pop("mooncake-connector-config", {})
        overlap = MANAGED_ARGS.intersection(_canonical_arg_key(key) for key in config)
        if overlap:
            raise ValueError(f"ATOM config cannot override srtctl-managed argument(s): {sorted(overlap)}")

        command = ["env", f"ATOM_HOST_IP={worker_ip}", *(nsys_prefix or [])]
        command.extend(
            frontend.worker_entrypoint(process, runtime) or ["python3", "-m", "atom.entrypoints.openai_server"]
        )
        command.extend(
            [
                "--model",
                runtime.worker_model_arg,
                "--host",
                "0.0.0.0",
                "--server-port",
                str(process.http_port),
                "-tp",
                str(_tensor_parallel_size(config, len(process.gpu_indices))),
            ]
        )
        kv_transfer_config = self._kv_transfer_config(process, worker_ip, extra_connectors, mooncake_config)
        if kv_transfer_config is not None:
            command.extend(["--kv-transfer-config", kv_transfer_config])
        command.extend(_config_to_cli_args(config))
        return command


def _config_to_cli_args(config: dict[str, Any]) -> list[str]:
    """Preserve ATOM's native CLI spelling, which mixes hyphens and underscores."""
    args: list[str] = []
    for key, value in sorted(config.items()):
        flag = f"--{key}"
        if value is True:
            args.append(flag)
        elif value is False or value is None:
            continue
        elif isinstance(value, list):
            args.extend([flag, *(str(item) for item in value)])
        else:
            args.extend([flag, str(value)])
    return args


def _canonical_arg_key(key: str) -> str:
    return key.lstrip("-").replace("_", "-")


def _tensor_parallel_size(config: dict[str, Any], gpus: int) -> int:
    """The -tp that fills the worker's GPUs once the role's other parallel sizes are taken out.

    ATOM runs dp x pp engines of tp x pcp GPUs each. Decode context parallelism
    reuses the TP GPUs and DP-attention turns -tp into DP ranks, so neither enters.
    """
    others = 1
    for key, value in config.items():
        name = _canonical_arg_key(key)
        if name in _WORLD_SIZE_ARGS:
            others *= int(value)
        elif any(arg.startswith(name) for arg in _WORLD_SIZE_ARGS):
            raise ValueError(f"ATOM config key {key!r} abbreviates a parallel size; spell it out")
    if others < 1 or gpus % others:
        raise ValueError(f"ATOM worker has {gpus} GPU(s), not a multiple of its pp x pcp x dp = {others}")
    return gpus // others
