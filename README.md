# rola-bench

The RoLA **benchmark suite** — two benchmarks that share one orchestration entrypoint. It defines
*what* to run; the generic [`fleet`](https://github.com/Blakexx/fleet-dev) package owns *how* (rent
GPUs, control over HTTP, sync results, idle-free teardown).

```
rola-bench/
  launch.py                 # orchestration: `python launch.py {mqar|lm} --boxes N`
  rola_bench/
    mqar/                   # MQAR benchmark — the canonical grid sweep (LIVE)
      experiments/  job.py  run_rla_sweep.py  verify.py
    lm/                     # language-modeling benchmark (SCAFFOLD — see lm/job.py)
      rola_hf/  train_lm.py  eval_lm.py  flame/  job.py
  docker/  mqar.Dockerfile  lm.Dockerfile  build.sh
```

Each benchmark module exports a `fleet.Job` (`rola_bench.mqar.job:mqar_job`,
`rola_bench.lm.job:lm_job`); `launch.py` looks one up by name and runs `Dispatcher(job)`.

## Dependencies

```
rola-bench ──▶ rola        (model + kernels)
           ──▶ fla-rola    (routed FLA fork)
           ──▶ zoology     (MQAR baselines)        [mqar only]
           ──▶ fleet       (GPU orchestration)
           ──▶ fla-core, transformers, …           (third-party stack)
```

No `sys.path` hacks, no vendored copies — real package edges. Configs reference
`zoology.mixers.rola.RoLAMixer` (rola is a first-class zoology mixer).

## Run the MQAR sweep

```bash
# 1. build images (base once, then the plug-in) — build → push → nuke-local
../fleet/docker/build_base.sh                 # blakeresearch/fleet-base:v1
docker/build.sh mqar --push --prune           # blakeresearch/rola-bench:v1

# 2. dispatch (reuses any run_id already complete in fleet_results/)
python launch.py mqar --boxes 6
```

Results land in `fleet_results/` (gitignored — operational state, not code). The dispatcher reuses
completed cells, requeues a dead box's owed cells, and tears every box down on drain.

## Images

| Image | Dockerfile | Benchmark |
|-------|------------|-----------|
| `blakeresearch/rola-bench:v1` | `docker/mqar.Dockerfile` | MQAR grid sweep |
| `blakeresearch/rola-lm:v1`    | `docker/lm.Dockerfile`   | LM (scaffold) |

Both build `FROM blakeresearch/fleet-base` and set `FLEET_RUN_CMD` so `fleet.box_server` knows how
to run a batch. `docker/build.sh <target> [--push] [--prune]` stages clean sources (no
`.git`/data/results) → lean, publication-safe.
