"""RoLA named instances — preset name -> zoology.mixers.rola.RoLAMixer kwargs.

The MQAR configs pass these to the mixer (which maps them onto fla_rola.layers.RoLA). The inner
kernel (rla / gla_scalar) is the only parameterized axis; tie_routers => symmetric (one router).
(Relocated from the now-deleted CLA rola.py — fla_rola is the model source of truth.)
"""


def rola_instance(name, d_qk, d_v, num_chunks, n_heads=4, routing='flat'):
    """RoLAMixer kwargs for a named instance. `routing` ∈ {flat,square,tree} selects the per-head
    factor topology (default 'flat' = the dense router, backward-compatible)."""
    common = dict(d_qk=d_qk, d_v=d_v, num_chunks=num_chunks, n_heads=n_heads, routing=routing,
                  use_short_conv=False)
    # RLA family (phi = feature map; global/per_state/kappa norm).
    if name == 'rola-rla-asym':                              # global-norm RLA (explicit; was default)
        return dict(kernel='rla', phi='elu', tie_routers=False,
                    kernel_kwargs={'state_norm': 'global'}, **common)
    if name == 'rola-rla-sym':
        return dict(kernel='rla', phi='elu', tie_routers=True,
                    kernel_kwargs={'state_norm': 'global'}, **common)
    if name == 'rola-rla-asym-ps':
        return dict(kernel='rla', phi='elu', tie_routers=False,
                    kernel_kwargs={'state_norm': 'per_state'}, **common)
    if name == 'rola-rla-sym-ps':
        return dict(kernel='rla', phi='elu', tie_routers=True,
                    kernel_kwargs={'state_norm': 'per_state'}, **common)
    if name == 'rola-rla-kappa-asym':
        return dict(kernel='rla', phi='elu', tie_routers=False,
                    kernel_kwargs={'state_norm': 'kappa'}, **common)
    if name == 'rola-rla-kappa-sym':
        return dict(kernel='rla', phi='elu', tie_routers=True,
                    kernel_kwargs={'state_norm': 'kappa'}, **common)
    if name == 'rola-rla-asym-tieinit':
        return dict(kernel='rla', phi='elu', tie_routers=False, tie_router_init=True,
                    kernel_kwargs={'state_norm': 'global'}, **common)
    # scalar-GLA family (optimized shared-gram).
    if name.startswith('rola-gla-scalar'):
        return dict(kernel='gla_scalar', tie_routers=name.endswith('-sym'),
                    kernel_kwargs={'normalized': '-norm-' in name}, **common)
    if name.startswith('rola-gla-kappa'):
        return dict(kernel='gla_scalar', tie_routers=name.endswith('-sym'),
                    kernel_kwargs={'state_norm': 'kappa'}, **common)
    if name.startswith('rola-gla-ps'):
        return dict(kernel='gla_scalar', tie_routers=name.endswith('-sym'),
                    kernel_kwargs={'state_norm': 'per_state'}, **common)
    raise ValueError(f"unknown RoLA instance: {name!r}")


ROLA_INSTANCES = ('rola-rla-asym', 'rola-rla-sym', 'rola-rla-asym-ps', 'rola-rla-sym-ps',
                  'rola-rla-kappa-asym', 'rola-rla-kappa-sym', 'rola-rla-asym-tieinit',
                  'rola-gla-scalar-sym', 'rola-gla-scalar-norm-sym',
                  'rola-gla-scalar-asym', 'rola-gla-scalar-norm-asym',
                  'rola-gla-kappa-sym', 'rola-gla-kappa-asym')
