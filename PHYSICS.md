# Frozen physics constants

**`boids.py` is authoritative.** This file is generated from `Physics()` and is
a convenience for reading, never a place to change a value — editing it changes
nothing and would recreate the exact divergence commit `79d9a03` removed.

Regenerate with:

```
python -c "from dataclasses import fields; from boids import Physics; p=Physics(); \
[print(f.name, getattr(p, f.name)) for f in fields(p)]"
```

Generated at commit `79d9a03`.

## The constants

| name | value | meaning |
|---|---|---|
| `dt` | 0.1 | seconds per tick, fixed |
| `bounds` | 100.0 | the world is a bounds x bounds torus |
| `max_speed` | 5.0 | speed ceiling; velocities above it are rescaled down |
| `min_speed` | 2.0 | speed floor / cruise speed; velocities below it are rescaled up |
| `separation_radius` | 20.0 | neighbours closer than this push a bird away |
| `alignment_radius` | 30.0 | neighbours closer than this contribute a heading |
| `cohesion_radius` | 40.0 | neighbours closer than this pull a bird toward their centre |
| `sep_force_cap` | 1.0 | per-neighbour ceiling on the 1/d repulsion, applied before averaging |
| `w_sep` | 1.5 | weight on the recorded separation force |
| `w_ali` | 1.0 | weight on the recorded alignment force |
| `w_coh` | 0.3 | weight on the recorded cohesion force |

`Physics` is a frozen dataclass, so an instance cannot be mutated mid-rollout.
`__post_init__` rejects non-positive `dt`, `bounds` or `max_speed`, a
`min_speed` outside `[0, max_speed]`, and any radius outside `(0, bounds/2)` —
past `bounds/2` the minimum-image convention stops distinguishing near from far
and the local rules quietly become global.

`min_speed = 0.0` means no floor, and is legal; `min_speed == max_speed` is the
constant-speed flocking model, and is also legal.

## What these values imply

Measured under exactly the constants above, with the parameters of each
measurement stated — no figure here is from a single seed.

| quantity | value | measured over |
|---|---|---|
| distance covered per tick at cruise | 0.2 | `min_speed * dt` |
| ticks to cross the separation radius | 100 | `separation_radius / (min_speed * dt)` |
| largest possible torus separation | 70.7 | `(bounds/2) * sqrt(2)` |
| flock orders itself (polarisation > 0.99) | median tick 130, p90 213 | N=12, 40 seeds, 800 ticks |
| the same at higher density | median tick 94, p90 140 | N=30, 40 seeds, 800 ticks |
| mean neighbours in sep / ali / coh | 1.5 / 6.2 / 7.0 of 11 | N=12, seed 0, tick 300 |
| pairwise label correlation, worst | -0.574 (mean -0.254) | N=12, 30 seeds, ticks 0-150 |
| speed exactly zero | only when a bird has no neighbour in any radius *and* started at rest; never at N >= 10 | 60,000 bird-ticks across densities and both start conditions |

Alignment and cohesion see nearly the whole flock at N=8-12. That is deliberate,
accepted on the evidence that the three labels stay distinct anyway. See
`NOTES.md`.
