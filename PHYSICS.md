# Frozen physics constants

**`boids.py` is authoritative.** This file is generated from `Physics()` and is
a convenience for reading, never a place to change a value — editing it changes
nothing and would recreate the exact divergence commit `dd85096` removed.

Regenerate with:

```
python -c "from dataclasses import fields; from boids import Physics; p=Physics(); \
[print(f.name, getattr(p, f.name)) for f in fields(p)]"
```

Generated after commit `cd30037`, with soft edges and the force cap in place.

## The constants

| name | value | meaning |
|---|---|---|
| `dt` | 0.1 | seconds per tick, fixed |
| `bounds` | 100.0 | the world is a bounds x bounds torus |
| `max_speed` | 5.0 | speed ceiling; velocities above it are rescaled down |
| `min_speed` | 1.0 | speed floor / cruise speed; velocities below it are rescaled up |
| `separation_radius` | 20.0 | neighbours closer than this push a bird away |
| `alignment_radius` | 30.0 | neighbours closer than this contribute a heading |
| `cohesion_radius` | 40.0 | neighbours closer than this pull a bird toward their centre |
| `sep_force_cap` | 1.0 | per-neighbour ceiling on the 1/d repulsion, applied before averaging |
| `edge_softness` | 0.25 | share of each radius over which a neighbour's weight fades from 1 to 0 |
| `max_force` | 0.3 | ceiling on the length of each weighted rule force; `None` means no cap |
| `w_sep` | 1.5 | weight on the recorded separation force |
| `w_ali` | 1.0 | weight on the recorded alignment force |
| `w_coh` | 0.3 | weight on the recorded cohesion force |

`Physics` is a frozen dataclass, so an instance cannot be mutated mid-rollout.
`__post_init__` rejects non-positive `dt`, `bounds` or `max_speed`, a
`min_speed` outside `[0, max_speed]`, and any radius outside `(0, bounds/2)` —
past `bounds/2` the minimum-image convention stops distinguishing near from far
and the local rules quietly become global.

`min_speed = 0.0` means no floor, and is legal; `min_speed == max_speed` is the
constant-speed flocking model, and is also legal. `edge_softness` must lie in
`[0, 1]` and `max_force` must be positive or `None`.

## Soft edges and the force cap

sim-v1 counted a neighbour fully up to the radius and not at all past it, so a
bird crossing a radius changed its neighbour's force in a single jump. Those
jumps cannot be predicted from the state a tick earlier, and they made settled
flight unlearnable.

**Soft edges.** Each neighbour now has a weight: 1 up to
`(1 - edge_softness) * radius`, falling in a straight line to 0 at the radius.
Every rule takes a weighted mean, divided by `max(sum of weights, 1)` rather
than by the sum alone. Dividing by the sum alone would let a lone neighbour
push at full strength right up to the radius and then vanish, which is the
jump this removes. Alignment subtracts `min(sum of weights, 1) * own velocity`
for the same reason. With `edge_softness = 0` every weight is 1 and the rules
are sim-v1's.

**The force cap.** Each rule's force is weighted, then shrunk to length
`max_force` if it is longer, per bird and per rule, not on the total. Without
it alignment dominated the early, disordered ticks and separation and cohesion
barely moved the birds. The recorded forces are the capped ones, which are the
ones added to the velocity: recorded after the force cap, before the speed
cap.

## What these values imply

The first three rows follow from the constants above. The rest were measured on
2026-09-01 in the sim-v1 world (hard edges, no force cap, `min_speed` still
2.0), with the parameters of each measurement stated. No figure here is from a
single seed. They have not been remeasured with soft edges and the cap; the
settle times in particular are longer now.

| quantity | value | measured over |
|---|---|---|
| distance covered per tick at cruise | 0.1 | `min_speed * dt` |
| ticks to cross the separation radius | 200 | `separation_radius / (min_speed * dt)` |
| largest possible torus separation | 70.7 | `(bounds/2) * sqrt(2)` |
| flock orders itself (polarisation > 0.99) | median tick 130, p90 213 | N=12, 40 seeds, 800 ticks |
| the same at higher density | median tick 94, p90 140 | N=30, 40 seeds, 800 ticks |
| mean neighbours in sep / ali / coh | 1.5 / 6.2 / 7.0 of 11 | N=12, seed 0, tick 300 |
| pairwise label correlation, worst | -0.574 (mean -0.254) | N=12, 30 seeds, ticks 0-150 |
| speed exactly zero | only when a bird has no neighbour in any radius *and* started at rest; never at N >= 10 | 60,000 bird-ticks across densities and both start conditions |

Alignment and cohesion see nearly the whole flock at N=8-12. That is deliberate,
accepted on the evidence that the three labels stay distinct anyway. 
