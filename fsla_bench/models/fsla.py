"""THE FSLA ARMS: every FSLA cell fsla-bench measures, as fsla's own objects, built only through fla.

A WIRING is one of fsla's level spellings per routing level, outermost first (`fsla.dense_routing`,
`fsla.split_routing`, `fsla.union_routing`, `fsla.tied_routing`), each taking the level's width. A CELL is a wiring at
a state count N: every level's width (the uniform factorization b**D = N unless widths are given), its levels as fsla
builds them, the decay source as fsla's JSON (`None`, or `{"type": "LearnedDecay", ...}`; `fla.layers.fsla.decode`
builds it with the levels' widths), and the route projection's `bias`. Every bench builds a cell one way --
`mixer_config` for zoology's MQAR model, `layer` for a bare fla layer -- and reads a built cell's state back off the
layer (`state_floats`), never from a formula.

What fsla refuses (a configuration outside its built envelope, prefill while its kernel is rebuilt, training before its
native backward) refuses through every one of these builds.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import fsla

#: The wirings. The first four are the canonical MQAR slate; the rest are the variants a grid axis asks for (the alpha-2
#: write split, the tied spelling, the depth hybrids).
WIRINGS: dict[str, tuple[Callable[[int], object], ...]] = {
    "fsla-arm1-densread-sparsewrite": (partial(fsla.split_routing, alpha=1.5, sparse_duty="write"), fsla.dense_routing),
    "fsla-arm2-union": (partial(fsla.union_routing, alpha=1.5), fsla.dense_routing),
    "fsla-arm3-levelsplit": (partial(fsla.split_routing, alpha=1.5, sparse_duty="write"),
                             partial(fsla.split_routing, alpha=1.5, sparse_duty="read")),
    "fsla-d1-dense": (fsla.dense_routing,),
    "fsla-arm3-levelsplit-a2w": (partial(fsla.split_routing, alpha=2.0, sparse_duty="write"),
                                 partial(fsla.split_routing, alpha=1.5, sparse_duty="read")),
    "fsla-d1-dense-tied": (partial(fsla.tied_routing, activation=fsla.softmax()),),
    "fsla-hybrid": (partial(fsla.union_routing, alpha=1.5), partial(fsla.split_routing, alpha=1.5, sparse_duty="write")),
    "fsla-hybrid-tiedtop": (partial(fsla.tied_routing, activation=fsla.entmax(1.5)),
                            partial(fsla.split_routing, alpha=1.5, sparse_duty="write")),
}

CANONICAL_WIRINGS = ("fsla-arm1-densread-sparsewrite", "fsla-arm2-union", "fsla-arm3-levelsplit", "fsla-d1-dense")


class NoSpelling(ValueError):
    """A wiring has no spelling at this state count (a D-level wiring at an N with no uniform b**D = N)."""


def uniform_widths(depth: int, n: int) -> tuple[int, ...]:
    """The widths (b,) * depth with b**depth == n, or NoSpelling."""
    b = round(n ** (1.0 / depth))
    for candidate in (b - 1, b, b + 1):
        if candidate >= 2 and candidate ** depth == n:
            return (candidate,) * depth
    raise NoSpelling(f"N={n} has no uniform {depth}-level factorization b**{depth}")


@dataclass(frozen=True)
class Cell:
    """A wiring at concrete widths, with its decay source and the route projection's bias."""

    wiring: str
    widths: tuple[int, ...]
    decay: dict | None = None
    bias: bool = True
    gain_bias_init: float | None = None
    levels: tuple = field(init=False)

    def __post_init__(self) -> None:
        if self.wiring not in WIRINGS:
            raise ValueError(f"unknown FSLA wiring {self.wiring!r}; expected one of {sorted(WIRINGS)}")
        spellings = WIRINGS[self.wiring]
        if len(self.widths) != len(spellings):
            raise ValueError(f"{self.wiring} has {len(spellings)} level(s); widths {self.widths} have {len(self.widths)}")
        object.__setattr__(self, "levels", tuple(spell(int(w)) for spell, w in zip(spellings, self.widths, strict=True)))

    @property
    def n(self) -> int:
        return math.prod(self.widths)

    def layer_kwargs(self) -> dict:
        """fla.layers.FSLA's routing arguments, fsla's objects as their JSON so a config can carry them."""
        from fla.layers.fsla import encode

        return {"levels": encode(list(self.levels)), "decay": self.decay, "bias": self.bias,
                "gain_bias_init": self.gain_bias_init}


def cell(wiring: str, n: int, *, widths=None, decay: dict | None = None, bias: bool = True) -> Cell:
    """`wiring` at state count `n`: the given widths (which must multiply to n) or the uniform factorization."""
    depth = len(WIRINGS[wiring]) if wiring in WIRINGS else 0
    if widths is None:
        widths = (int(n),) if depth == 1 else uniform_widths(depth, int(n))
    elif math.prod(widths) != int(n):
        raise ValueError(f"widths {list(widths)} multiply to {math.prod(widths)}, not N={n}")
    return Cell(wiring, tuple(int(w) for w in widths), decay=decay, bias=bias)


def layer(c: Cell, *, hidden_size: int, num_heads: int, d_v: int, layer_idx: int = 0):
    """The cell as a bare flan FSLA layer."""
    from fla.layers.fsla import FSLA

    return FSLA(hidden_size=hidden_size, num_heads=num_heads, d_v=d_v, layer_idx=layer_idx, **c.layer_kwargs())


def mixer_config(c: Cell, *, n_heads: int, d_v: int) -> dict:
    """The cell as zoology's sequence-mixer config (`zoology.mixers.fsla.FSLAMixer` over fla)."""
    return {"name": "zoology.mixers.fsla.FSLAMixer", "kwargs": {"n_heads": n_heads, "d_v": d_v, **c.layer_kwargs()}}


def state_floats(module) -> tuple[int, int]:
    """(content, overhead) recurrent floats per layer, read off a built FSLA layer or mixer: content is H * N * d_v,
    the matched capacity axis; overhead is the rest (the mass column), reported beside it."""
    stats = module.get_stats() if hasattr(module, "get_stats") else module.layer.get_stats()
    content = stats["n_heads"] * stats["num_chunks"] * stats["d_v"]
    return content, stats["state_floats"] - content
