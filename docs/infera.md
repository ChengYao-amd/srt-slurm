# Infera Router

`frontend.type: infera` runs the [Infera](https://github.com/AMD-AGI/Infera) router in
front of native ATOM workers (`engine: atom`). Infera workers register themselves in
etcd, so this is a dynamic frontend like Dynamo: srtctl starts etcd, launches each
worker through Infera's ATOM launcher, and starts the router with no worker list.

Upstream references below are pinned to Infera
[`34e84c66`](https://github.com/AMD-AGI/Infera/tree/34e84c6609df74f61e82277b4693134ba3033f9c).

## What srtctl launches

| Process | Command |
| --- | --- |
| etcd | the implied `etcd` service on the infra node, as for Dynamo |
| each worker | `env ATOM_HOST_IP=<worker ip> env INFERA_READINESS_PORT=<sys port> python3 -m infera.engine.atom --etcd-endpoint http://<infra>:2379 --advertise-host <worker ip>` followed by the usual ATOM server arguments |
| router | `python3 -m infera.server --host 0.0.0.0 --port <frontend port> --discovery-backend etcd --etcd-endpoint http://<infra>:2379 --request-transport http --router-tokenizer-path <model>` followed by `frontend.args` |

`infera.engine.atom` starts `atom.entrypoints.openai_server` with the arguments it does
not consume, waits for the server's `/health`, and then registers it in etcd with the
role read from `--kv-transfer-config` (`kv_producer` is prefill, `kv_consumer` is decode,
no role is `mixed`). After registering it opens a readiness listener on
`INFERA_READINESS_PORT`; srtctl points that at the worker's allocated system port so
colocated workers do not share Infera's default, 30090
([`readiness.py`](https://github.com/AMD-AGI/Infera/blob/34e84c6609df74f61e82277b4693134ba3033f9c/infera/engine/readiness.py#L37-L42)).

Workers and the router run the recipe's `setup_script` first. Workers use the
`model.container` image; the router uses `frontend.container_image`, else
`model.container`. Each image, or the setup script, must provide Infera, and the
worker image must also provide ATOM.

Readiness polls the router's `GET /v1/workers` and counts `active` registrations by
`disagg_mode`; an aggregate worker registers as `mixed` and counts as decode. The
router's `/health` is not used: it answers 200 whatever is registered
([`app.py`](https://github.com/AMD-AGI/Infera/blob/34e84c6609df74f61e82277b4693134ba3033f9c/infera/server/app.py#L292-L310),
[`/health`](https://github.com/AMD-AGI/Infera/blob/34e84c6609df74f61e82277b4693134ba3033f9c/infera/server/app.py#L688-L691)).
Worker metrics, benchmark endpoints and profiler control use each ATOM server's own
HTTP port.

## Recipe

See the [one-node prefill/decode example](https://github.com/NVIDIA/srt-slurm/blob/main/examples/atom/infera-1node-pd.yaml).
Router policy flags go in `frontend.args`, verbatim:

```yaml
frontend:
  type: infera
  enable_multiple_frontends: false
  args:
    router-backend: python
    router-policy: round-robin
    kv-event-transport: zmq
```

Engine flags go under `roles.<role>.args` exactly as for [ATOM with AToMesh](engines.md#atom-with-atomesh);
the ATOM backend builds the same server arguments, including `-tp` from the parallel
sizes and the Mooncake settings, for both frontends.

Configuration loading rejects:

- `frontend.args` that srtctl sets (`host`, `port`, `discovery-backend`,
  `etcd-endpoint`, `request-transport`, `router-tokenizer-path`) or a prefix of one,
  which the router's argument parser would accept as that flag.
- `kv-event-transport: nats`: the job runs no NATS server.
- `router-backend: rust` with prefill/decode roles: Infera's Rust router has no ATOM
  Mooncake hand-off
  ([`protocol.rs`](https://github.com/AMD-AGI/Infera/blob/34e84c6609df74f61e82277b4693134ba3033f9c/rust/router/src/protocol.rs#L29-L44)).
- `services[etcd].external`: workers and router use the job's own etcd.
- Role args that set or abbreviate an argument srtctl manages (`model`, `host`,
  `server-port`, `tensor-parallel-size`, `kv-transfer-config`, `etcd-endpoint`,
  `advertise-host`). ATOM accepts an abbreviation, but the launcher reads `-tp` and
  `--kv-transfer-config` with abbreviations disabled
  ([`args.py`](https://github.com/AMD-AGI/Infera/blob/34e84c6609df74f61e82277b4693134ba3033f9c/infera/engine/atom/args.py#L164-L211)),
  so the registration and the server would disagree.
- `served-model-name` in role args, or a prefix of it: the launcher registers the
  worker under its `--model`
  ([`__main__.py`](https://github.com/AMD-AGI/Infera/blob/34e84c6609df74f61e82277b4693134ba3033f9c/infera/engine/atom/__main__.py#L125)),
  and the router routes a request only to workers registered under its `model`
  ([`worker_pool.py`](https://github.com/AMD-AGI/Infera/blob/34e84c6609df74f61e82277b4693134ba3033f9c/infera/common/worker_pool.py#L237-L238)).
- `enable-kv-events` in role args: the launcher's KV-event publisher binds a port
  srtctl does not allocate
  ([`__main__.py`](https://github.com/AMD-AGI/Infera/blob/34e84c6609df74f61e82277b4693134ba3033f9c/infera/engine/atom/__main__.py#L87-L99)).

## Limits

Each worker fits on one node, as for every ATOM recipe. Workers do not publish KV
events, so `router-policy: kv-aware` has no cache view to route by; use
`round-robin`, optionally with `session-affinity`.
