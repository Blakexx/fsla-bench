"""RoLA named instances — preset name -> zoology.mixers.rola.RoLAMixer kwargs.

The MQAR configs pass these to the mixer (which maps them onto fla_rola.layers.RoLA). The inner
kernel (rla / gla_scalar) is the only parameterized axis; tie_routers => symmetric (one router).
(Relocated from the now-deleted CLA rola.py — fla_rola is the model source of truth.)

`rola_instance`'s OWN inputs keep the `num_chunks`/`routing`∈{flat,square,tree} spelling — the
convenient state-count-ladder + topology-name vocabulary used throughout the MQAR grid builders
(`rola_bench.mqar.build_configs`, `rola_bench.suites.mqar`) — but the returned dict speaks
`zoology.mixers.rola.RoLAMixer`'s NEW structural surface directly (`levels_per_head`/
`states_per_level`/`level_routing_configs`; #config-refactor 2026-07-06), via the shared
`rola_bench.rola_compat` translators. No shim lives in the mixer itself anymore.
"""
from rola_bench.rola_compat import level_routing_config_from_tie_routers, topology_from_legacy


def rola_instance(name, d_qk, d_v, num_chunks, n_heads=4, routing='flat'):
    """RoLAMixer kwargs for a named instance. `routing` ∈ {flat,square,tree} selects the per-head
    factor topology (default 'flat' = the dense router, backward-compatible)."""
    D, b = topology_from_legacy(num_chunks, routing)
    common = dict(d_qk=d_qk, d_v=d_v, levels_per_head=D, states_per_level=b, n_heads=n_heads,
                  use_short_conv=False)

    def _routed(tie_routers, **extra):
        return dict(level_routing_configs=level_routing_config_from_tie_routers(tie_routers),
                    **extra, **common)

    # RLA family (phi = feature map; global/per_state/kappa norm).
    if name == 'rola-rla-asym':                              # global-norm RLA (explicit; was default)
        return _routed(False, kernel='rla', phi='elu', kernel_kwargs={'state_norm': 'global'})
    if name == 'rola-rla-sym':
        return _routed(True, kernel='rla', phi='elu', kernel_kwargs={'state_norm': 'global'})
    if name == 'rola-rla-asym-ps':
        return _routed(False, kernel='rla', phi='elu', kernel_kwargs={'state_norm': 'per_state'})
    if name == 'rola-rla-sym-ps':
        return _routed(True, kernel='rla', phi='elu', kernel_kwargs={'state_norm': 'per_state'})
    if name == 'rola-rla-kappa-asym':
        return _routed(False, kernel='rla', phi='elu', kernel_kwargs={'state_norm': 'kappa'})
    if name == 'rola-rla-kappa-sym':
        return _routed(True, kernel='rla', phi='elu', kernel_kwargs={'state_norm': 'kappa'})
    if name == 'rola-rla-asym-tieinit':
        return _routed(False, kernel='rla', phi='elu', tie_router_init=True,
                       kernel_kwargs={'state_norm': 'global'})
    # scalar-GLA family (optimized shared-gram).
    if name.startswith('rola-gla-scalar'):
        return _routed(name.endswith('-sym'), kernel='gla_scalar',
                       kernel_kwargs={'normalized': '-norm-' in name})
    if name.startswith('rola-gla-kappa'):
        return _routed(name.endswith('-sym'), kernel='gla_scalar', kernel_kwargs={'state_norm': 'kappa'})
    if name.startswith('rola-gla-ps'):
        return _routed(name.endswith('-sym'), kernel='gla_scalar', kernel_kwargs={'state_norm': 'per_state'})
    raise ValueError(f"unknown RoLA instance: {name!r}")


ROLA_INSTANCES = ('rola-rla-asym', 'rola-rla-sym', 'rola-rla-asym-ps', 'rola-rla-sym-ps',
                  'rola-rla-kappa-asym', 'rola-rla-kappa-sym', 'rola-rla-asym-tieinit',
                  'rola-gla-scalar-sym', 'rola-gla-scalar-norm-sym',
                  'rola-gla-scalar-asym', 'rola-gla-scalar-norm-asym',
                  'rola-gla-kappa-sym', 'rola-gla-kappa-asym')
