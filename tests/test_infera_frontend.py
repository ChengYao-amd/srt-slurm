# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Infera router frontend (`frontend.type: infera`) over native ATOM workers."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from marshmallow import ValidationError

from srtctl.backends import AtomBackend
from srtctl.core.config import resolve_config_with_defaults
from srtctl.core.schema import RoleConfig, SrtConfig
from srtctl.core.topology import Process
from srtctl.frontends import InferaFrontend, get_frontend
from srtctl.services.implicit import effective_services

WORKER_IP = "10.0.0.20"
AGG = {"agg": {"nodes": 1, "workers": 1, "gpus": 4}}
PD = {
    "prefill": {"nodes": 1, "workers": 1, "gpus": 4, "args": {"pipeline-parallel-size": 4}},
    "decode": {"nodes": "colocate", "workers": 1, "gpus": 4, "args": {"decode-context-parallel-size": 4}},
}


def _recipe(roles: dict, args: dict | None = None, **extra) -> dict:
    return {
        "schema": 2,
        "engine": "atom",
        "name": "infera",
        "model": {"path": "hf:Qwen/Qwen3-0.6B", "container": "infera-atom:latest", "precision": "bf16"},
        "resources": {"gpu_type": "mi355x", "gpus_per_node": 8},
        "roles": roles,
        "frontend": {"type": "infera", "enable_multiple_frontends": False, "args": args or {}},
        **extra,
    }


def _load(data: dict) -> SrtConfig:
    return SrtConfig.Schema().load(resolve_config_with_defaults(data, None))


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(worker_model_arg="/model", network_interface="eno0", infra_node_ip="10.0.0.1")


def _build(backend: AtomBackend, process: Process) -> list[str]:
    with (
        patch("srtctl.core.slurm.get_hostname_ip", return_value=WORKER_IP),
        patch("srtctl.frontends.infera.get_hostname_ip", return_value=WORKER_IP),
    ):
        return backend.build_worker_command(process, [process], _runtime(), frontend_type="infera")


def test_registry_resolves_infera_for_atom() -> None:
    frontend = get_frontend("infera")
    assert isinstance(frontend, InferaFrontend)
    assert frontend.required_backend == "atom"
    assert frontend.worker_launch == "direct"
    assert frontend.worker_api_port("agg") == "allocated"


def test_infera_requires_atom_backend() -> None:
    data = _recipe(AGG)
    data["engine"] = "sglang"
    with pytest.raises(ValidationError, match="frontend.type: infera requires backend.type: atom"):
        _load(data)


def test_infera_implies_etcd_on_the_infra_node_and_no_nats() -> None:
    services = effective_services(_load(_recipe(AGG)))
    etcd = [entry for entry in services if entry.service.type == "etcd"]
    assert len(etcd) == 1 and etcd[0].implicit and etcd[0].reason == "frontend.type infera"
    assert not [entry for entry in services if entry.service.type == "nats"]


def test_aggregate_worker_runs_atom_under_the_infera_launcher() -> None:
    backend = AtomBackend(roles={"agg": RoleConfig(args={"enable-dp-attention": True})})
    process = Process("node0", frozenset(range(4)), 7500, 6100, "agg", 0, nixl_port=5400)

    assert _build(backend, process) == [
        "env",
        f"ATOM_HOST_IP={WORKER_IP}",
        "env",
        "INFERA_READINESS_PORT=7500",
        "python3",
        "-m",
        "infera.engine.atom",
        "--etcd-endpoint",
        "http://10.0.0.1:2379",
        "--advertise-host",
        WORKER_IP,
        "--model",
        "/model",
        "--host",
        "0.0.0.0",
        "--server-port",
        "6100",
        "-tp",
        "4",
        "--enable-dp-attention",
    ]


def test_pipeline_parallel_prefill_registers_as_a_mooncake_producer() -> None:
    """TP1 x PP4 on four GPUs with four Mooncake send workers, next to an offload connector."""
    args = {
        "pipeline-parallel-size": 4,
        "mooncake-connector-config": {"num_worker_threads": 4},
        "extra-kv-connectors": [{"kv_connector": "lmcache_offload", "kv_role": "offload"}],
    }
    backend = AtomBackend(roles={"prefill": RoleConfig(args=args)}, mooncake_protocol="rdma")
    process = Process("node0", frozenset(range(4)), 7500, 6100, "prefill", 0, nixl_port=5400)

    command = _build(backend, process)

    assert command[command.index("-tp") + 1] == "1"
    assert command[command.index("--pipeline-parallel-size") + 1] == "4"
    assert json.loads(command[command.index("--kv-transfer-config") + 1]) == {
        "kv_connector": "multi",
        "connectors": [
            {
                "kv_role": "kv_producer",
                "kv_connector": "mooncake",
                "proxy_ip": WORKER_IP,
                "handshake_port": 5400,
                "protocol": "rdma",
                "num_worker_threads": 4,
            },
            {"kv_connector": "lmcache_offload", "kv_role": "offload"},
        ],
    }


def test_colocated_prefill_and_decode_get_their_own_readiness_ports() -> None:
    config = _load(_recipe(PD))
    prefill, decode = config.backend.endpoints_to_processes(
        config.allocate_worker_endpoints(["node0"]), frontend_type="infera"
    )

    assert (sorted(prefill.gpu_indices), sorted(decode.gpu_indices)) == ([0, 1, 2, 3], [4, 5, 6, 7])
    assert "INFERA_READINESS_PORT=7500" in _build(config.backend, prefill)
    assert "INFERA_READINESS_PORT=7501" in _build(config.backend, decode)


def test_worker_ports_are_atoms_own_server_and_the_readiness_listener() -> None:
    frontend = InferaFrontend()
    leader = Process("node0", frozenset(range(4)), 7500, 6100, "prefill", 0)

    assert frontend.worker_metrics_port(leader, _runtime()) == 6100
    assert frontend.worker_endpoint_port(leader, None, _runtime()) == 6100
    assert frontend.worker_metrics_path(AtomBackend(), "prefill") == "/metrics"
    assert frontend.worker_ready_port(leader) == 7500


@pytest.mark.parametrize(
    ("roles", "args", "extra", "message"),
    [
        (AGG, {"port": 9000}, {}, r"frontend.args \['port'\] are managed by srtctl"),
        (AGG, {"etcd_endpoint": "http://x:2379"}, {}, r"frontend.args \['etcd-endpoint'\] are managed"),
        (AGG, {"request-trans": "nats"}, {}, r"frontend.args \['request-trans'\] are managed"),
        (AGG, {"kv-event-transport": "nats"}, {}, "runs no NATS server"),
        (PD, {"router-backend": "rust"}, {}, "prefill/decode needs router-backend: python"),
        (
            AGG,
            {},
            {"services": [{"name": "etcd", "type": "etcd", "external": "http://etcd.shared:2379"}]},
            r"drop services\[etcd\].external",
        ),
    ],
)
def test_validate_rejects_router_settings_srtctl_cannot_honor(
    roles: dict, args: dict, extra: dict, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        _load(_recipe(roles, args, **extra))


@pytest.mark.parametrize(
    ("key", "managed"),
    [
        ("tensor-parallel", "tensor-parallel"),
        ("kv_transfer_conf", "kv-transfer-conf"),
        ("advertise-host", "advertise-host"),
        ("--etcd-endpoint", "etcd-endpoint"),
    ],
)
def test_validate_rejects_worker_args_srtctl_manages_or_their_prefixes(key: str, managed: str) -> None:
    roles = {"agg": {**AGG["agg"], "args": {key: "x"}}}
    with pytest.raises(ValidationError, match=rf"roles.agg.args \['{managed}'\] are managed by srtctl"):
        _load(_recipe(roles))


def test_validate_rejects_kv_events_on_a_worker() -> None:
    roles = {**PD, "decode": {**PD["decode"], "args": {"enable_kv_events": True}}}
    with pytest.raises(ValidationError, match="roles.decode.args.enable-kv-events is not supported"):
        _load(_recipe(roles))


@pytest.mark.parametrize("key", ["served-model-name", "served_model"])
def test_validate_rejects_a_served_model_name_the_router_would_not_match(key: str) -> None:
    roles = {"agg": {**AGG["agg"], "args": {key: "glm"}}}
    with pytest.raises(ValidationError, match="roles.agg.args.served-model-name is not supported"):
        _load(_recipe(roles))


def test_validate_accepts_the_python_router_policy_flags_and_rust_for_aggregate() -> None:
    _load(_recipe(PD, {"router-backend": "python", "router-policy": "round-robin", "kv-event-transport": "zmq"}))
    _load(_recipe(AGG, {"router-backend": "rust"}))


@pytest.mark.parametrize(
    ("workers", "expected", "counted", "ready"),
    [
        ([{"disagg_mode": "mixed", "status": "active"}], (0, 1), (0, 1), True),
        (
            [{"disagg_mode": "prefill", "status": "active"}, {"disagg_mode": "decode", "status": "active"}],
            (1, 1),
            (1, 1),
            True,
        ),
        (
            [{"disagg_mode": "prefill", "status": "active"}, {"disagg_mode": "decode", "status": "draining"}],
            (1, 1),
            (1, 0),
            False,
        ),
        ([], (0, 1), (0, 0), False),
    ],
)
def test_readiness_counts_active_registrations_by_mode(
    workers: list, expected: tuple[int, int], counted: tuple[int, int], ready: bool
) -> None:
    result = InferaFrontend().parse_health({"workers": workers}, *expected)
    assert result.ready is ready
    assert (result.prefill_ready, result.decode_ready) == counted


def test_readiness_without_a_worker_list_is_not_ready() -> None:
    assert not InferaFrontend().parse_health({"status": "ok", "active_workers": 2}, 1, 1).ready


def test_router_command_watches_etcd_over_http(tmp_path: Path) -> None:
    topology = SimpleNamespace(frontend_nodes=["node0"], frontend_port=8000)
    runtime = SimpleNamespace(
        log_dir=tmp_path,
        infra_node_ip="10.0.0.1",
        worker_model_arg="/model",
        container_image="infera-atom:latest",
        container_mounts={},
        environment={"HF_HOME": "/hf"},
        srun_options={},
        nodes=SimpleNamespace(het_group_for=lambda node: None),
    )
    config = SimpleNamespace(
        frontend=SimpleNamespace(
            args={"router-policy": "round-robin", "session-affinity": "prefill"},
            env={"INFERA_LOG_LEVEL": "info"},
            container_image=None,
            numa_bind=False,
        ),
        setup_script="install-infera.sh",
    )
    with patch.object(InferaFrontend, "start_process", return_value=MagicMock()) as srun:
        InferaFrontend().start_frontends(topology, runtime, config, AtomBackend(), [])

    kwargs = srun.call_args.kwargs
    assert kwargs["command"] == [
        "python3",
        "-m",
        "infera.server",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--discovery-backend",
        "etcd",
        "--etcd-endpoint",
        "http://10.0.0.1:2379",
        "--request-transport",
        "http",
        "--router-tokenizer-path",
        "/model",
        "--router-policy",
        "round-robin",
        "--session-affinity",
        "prefill",
    ]
    assert kwargs["env_to_set"] == {"HF_HOME": "/hf", "INFERA_LOG_LEVEL": "info"}
    assert "install-infera.sh" in kwargs["bash_preamble"]
