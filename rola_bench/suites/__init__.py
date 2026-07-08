"""Suite entry points (DAG F1 benchmark-suite productionization): `python -m rola_bench.suites.lm`,
`.mqar`, `.perf`. Each declares its pinned config list + seeds + output schema and supports
`--smoke` (construct everything, run ONE tiny step/measurement, no training/full benchmark). See
`rola_bench/perf/METHODOLOGY.md` and the top-level README's suite table for where real results land.
"""
