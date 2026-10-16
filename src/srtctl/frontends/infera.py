# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Infera router frontend (`frontend.type: infera`) for native ATOM workers.

Infera (https://github.com/AMD-AGI/Infera) runs each worker through
``python3 -m infera.engine.atom``, which starts ATOM's own OpenAI server and then
registers it in etcd; the router (``python3 -m infera.server``) watches etcd and
lists what it routes to at ``GET /v1/workers``. Workers are therefore the engine's
own servers on their allocated ports, discovered rather than passed on the
router's command line.
"""

from __future__ import annotations

import logging
import shlex
import threading
from collections.abc import Collection
from typing import TYPE_CHECKING, Any, ClassVar

from srtctl.backends.atom import MANAGED_ARGS
from srtctl.core.health import WorkerHealthResult
from srtctl.core.slurm import get_hostname_ip, start_srun_process
from srtctl.frontends.base import numactl_prefix, register_frontend
from srtctl.frontends.dynamic_frontend import DynamicFrontend
from srtctl.frontends.static_router import setup_script_preamble
from srtctl.ports import ETCD_CLIENT_PORT
from srtctl.services.config import ServiceConfig
from srtctl.services.implicit import ETCD_SERVICE_NAME, EffectiveService, find_service, infra_placement

if TYPE_CHECKING:
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process

logger = logging.getLogger(__name__)

# Router flags derived from the allocation; a recipe value would point the router elsewhere.
MANAGED_ROUTER_ARGS = frozenset(
    {"host", "port", "discovery-backend", "etcd-endpoint", "router-tokenizer-path", "request-transport"}
)
# Worker launcher flags srtctl sets in ``worker_entrypoint``.
MANAGED_LAUNCHER_ARGS = frozenset({"etcd-endpoint", "advertise-host"})


def etcd_endpoint(runtime: RuntimeContext) -> str:
    """The implied etcd service on the infra node, as workers and router dial it."""
    return f"http://{runtime.infra_node_ip}:{ETCD_CLIENT_PORT}"


def _flag(key: Any) -> str:
    return str(key).lstrip("-").replace("_", "-")


def _reaches(key: str, flags: Collection[str]) -> bool:
    """Whether ``--key`` names one of ``flags``, in full or as an argparse prefix."""
    return any(flag.startswith(key) for flag in flags)


@register_frontend("infera")
class InferaFrontend(DynamicFrontend):
    """Infera's router in front of ATOM servers that register themselves over etcd."""

    type: ClassVar[str] = "infera"
    required_backend: ClassVar[str | None] = "atom"

    @property
    def health_endpoint(self) -> str:
        """Infera's ``/health`` answers 200 with no worker breakdown; ``/v1/workers`` lists each registration."""
        return "/v1/workers"

    def worker_entrypoint(self, process: Process, runtime: RuntimeContext) -> list[str] | None:
        """``infera.engine.atom`` registers the server it starts; its readiness listener is the allocated sys port."""
        return [
            "env",
            f"INFERA_READINESS_PORT={process.sys_port}",
            "python3",
            "-m",
            "infera.engine.atom",
            "--etcd-endpoint",
            etcd_endpoint(runtime),
            "--advertise-host",
            get_hostname_ip(process.node, runtime.network_interface),
        ]

    # A worker is ATOM's own OpenAI server: metrics, endpoint and profiler routes live on its HTTP port.
    def worker_metrics_port(self, process: Process, runtime: RuntimeContext) -> int | None:
        return process.http_port if process.is_leader and process.http_port > 0 else None

    def worker_endpoint_port(self, process: Process, config: Any, runtime: RuntimeContext) -> int | None:
        return process.http_port if process.is_leader and process.http_port > 0 else None

    def profiling_control_port(self, process: Process, config: Any, runtime: RuntimeContext) -> int | None:
        return process.http_port if process.http_port > 0 else None

    def worker_ready_port(self, process: Process) -> int:
        """INFERA_READINESS_PORT: open once the worker is registered (see ``worker_entrypoint``)."""
        return process.sys_port

    def implied_services(self, config: Any) -> list[EffectiveService]:
        """Workers register in etcd; the router watches it."""
        return [
            EffectiveService(
                ServiceConfig(name=ETCD_SERVICE_NAME, type="etcd", placement=infra_placement(config)),
                implicit=True,
                reason="frontend.type infera",
            )
        ]

    def validate(self, config: Any) -> None:
        """Keep srtctl's router and worker flags authoritative and the job free of NATS.

        ATOM's server and Infera's router accept an unambiguous prefix of a flag, so
        a prefix of a managed flag counts as that flag. Infera's worker launcher reads
        ``-tp`` and ``--kv-transfer-config`` with prefixes disabled, so a prefix would
        reach ATOM but not the registration the launcher writes.
        """
        args = {_flag(key): value for key, value in (config.frontend.args or {}).items()}
        managed = sorted(key for key in args if _reaches(key, MANAGED_ROUTER_ARGS))
        if managed:
            raise ValueError(f"frontend.args {managed} are managed by srtctl for frontend.type: infera")
        if any(_reaches(key, ("kv-event-transport",)) and value == "nats" for key, value in args.items()):
            raise ValueError("frontend.type: infera runs no NATS server; use kv-event-transport: zmq")
        rust = any(_reaches(key, ("router-backend",)) and value == "rust" for key, value in args.items())
        if rust and config.topology.is_disaggregated:
            raise ValueError(
                "frontend.type: infera prefill/decode needs router-backend: python; "
                "Infera's Rust router does not implement ATOM's Mooncake hand-off"
            )
        etcd = find_service(config, ETCD_SERVICE_NAME)
        if etcd is not None and etcd.external:
            raise ValueError(
                "frontend.type: infera workers register in the job's own etcd; drop services[etcd].external"
            )

        for mode in ("prefill", "decode", "agg"):
            keys = [_flag(key) for key in config.backend_for_role(mode).get_config_for_mode(mode)]
            managed = sorted(key for key in keys if _reaches(key, MANAGED_ARGS | MANAGED_LAUNCHER_ARGS))
            if managed:
                raise ValueError(f"roles.{mode}.args {managed} are managed by srtctl for frontend.type: infera")
            if "enable-kv-events" in keys:
                raise ValueError(
                    f"roles.{mode}.args.enable-kv-events is not supported with frontend.type: infera: "
                    "the KV-event publisher binds a port srtctl does not allocate"
                )
            if any(_reaches(key, ("served-model-name",)) for key in keys):
                raise ValueError(
                    f"roles.{mode}.args.served-model-name is not supported with frontend.type: infera: "
                    "Infera registers each worker under its --model, which clients must send"
                )

    def parse_health(self, response_json: dict, expected_prefill: int, expected_decode: int) -> WorkerHealthResult:
        """Count active registrations; an aggregate worker registers as ``mixed`` and counts as decode."""
        workers = response_json.get("workers")
        if not isinstance(workers, list):
            return WorkerHealthResult(ready=False, message=f"Key 'workers' not found in response: {response_json}")
        active = [w for w in workers if isinstance(w, dict) and w.get("status") == "active"]
        prefill = sum(w.get("disagg_mode") == "prefill" for w in active)
        decode = sum(w.get("disagg_mode") in ("decode", "mixed") for w in active)
        ready = prefill >= expected_prefill and decode >= expected_decode
        state = "ready" if ready else f"waiting for {expected_prefill}P + {expected_decode}D"
        return WorkerHealthResult(
            ready=ready,
            message=f"Infera {state}: {prefill} prefill and {decode} decode/mixed workers active",
            prefill_ready=prefill,
            prefill_expected=expected_prefill,
            decode_ready=decode,
            decode_expected=expected_decode,
        )

    def start_process(self, **kwargs: Any) -> Any:
        """Launch one router process; split out for adapter-specific tests."""
        return start_srun_process(**kwargs)

    def start_frontends(
        self,
        topology: Any,
        runtime: RuntimeContext,
        config: Any,
        backend: Any,
        backend_processes: list[Process],
        stop_event: threading.Event | None = None,
    ) -> list[ManagedProcess]:
        from srtctl.core.processes import FRONTEND_TERMINATE_TIMEOUT_SECONDS, ManagedProcess

        processes: list[ManagedProcess] = []
        for idx, node in enumerate(topology.frontend_nodes):
            # ATOM workers serve HTTP only, so requests go over HTTP and no NATS is needed.
            cmd = [
                "python3",
                "-m",
                "infera.server",
                "--host",
                "0.0.0.0",
                "--port",
                str(topology.frontend_port),
                "--discovery-backend",
                "etcd",
                "--etcd-endpoint",
                etcd_endpoint(runtime),
                "--request-transport",
                "http",
                "--router-tokenizer-path",
                runtime.worker_model_arg,
                *self.get_frontend_args_list(config.frontend.args),
            ]
            cmd = numactl_prefix(config) + cmd
            logger.info("Starting infera router %d on %s: %s", idx, node, shlex.join(cmd))
            log = runtime.log_dir / f"{node}_infera_{idx}.out"
            step_name = f"infera_{idx}"
            proc = self.start_process(
                command=cmd,
                nodelist=[node],
                output=str(log),
                container_image=config.frontend.container_image or str(runtime.container_image),
                container_mounts=runtime.container_mounts,
                env_to_set={**runtime.environment, **(config.frontend.env or {})} or None,
                bash_preamble=setup_script_preamble(config),
                het_group=runtime.nodes.het_group_for(node),
                step_name=step_name,
                srun_options=runtime.srun_options,
            )
            processes.append(
                ManagedProcess(
                    name=step_name,
                    popen=proc,
                    log_file=log,
                    node=node,
                    critical=True,
                    terminate_timeout=FRONTEND_TERMINATE_TIMEOUT_SECONDS,
                    step_name=step_name,
                )
            )
        return processes
