# The test suite

81 tests, ~6 seconds. `pytest -q` from the repo root.

These forces are the ground-truth labels for every probe and every ablation in
this project. A silent error in them corrupts every downstream result and
raises nothing, so this suite is written to catch errors that *look fine* —
plausible numbers, no exception, no NaN — rather than crashes.

Every test below has been checked by mutation: the bug it claims to catch was
introduced into `boids.py` deliberately, and the test failed. A test nobody has
watched fail is a test nobody has tested. Five of them were written because they
caught a real bug in real code, and those are marked.

| file | what it covers | tests |
|---|---|---|
| `test_geometry.py` | torus distance, the two-sided speed clamp, `Physics` validation | 21 |
| `test_separation.py` | `calculate_separation` | 10 |
| `test_alignment.py` | `calculate_alignment` | 6 |
| `test_cohesion.py` | `calculate_cohesion` | 11 |
| `test_step.py` | one tick: the integrator, the constants, the label invariants | 20 |
| `test_rollout.py` | hundreds of ticks: what only shows up over time | 13 |

`world.py` holds the shared constants and two baselines, so the six files cannot
drift into disagreeing about the size of the world. `PHYSICS` pins
`min_speed=0.0` — no floor — so that every test written before the floor existed
still means exactly what it meant then; `PHYSICS_CRUISING` is the same world with
the floor on. Tests that vary one constant use
`dataclasses.replace(PHYSICS, thing=...)`.

---

## test_geometry.py

The primitives every rule is built on.

| test | catches |
|---|---|
| `short_delta_passes_through_untouched` | a wrap applied when none is needed |
| `wraps_positive_delta` | `%` used without the `+ L/2 … − L/2` sandwich |
| `wraps_negative_delta` | sign errors in the wrap |
| `wrap_is_decided_per_axis` | one axis's wrap decision applied to both |
| `handles_position_wound_past_the_bounds` | a position 2.3 screens out of the world |
| `capper_leaves_slow_velocity_untouched` | a cap that always fires |
| `capper_shortens_the_arrow_without_turning_it` | `return max_speed` — a scalar broadcast into a vector slot, giving speed 7.07 and the wrong heading |
| `capper_leaves_velocity_exactly_at_the_cap` | an off-by-one on `>` vs `>=` |
| `capper_handles_a_motionless_bird` | division by zero speed → silent NaN |
| `floor_raises_a_slow_velocity_without_turning_it` | a floor that rotates instead of rescaling |
| `floor_leaves_a_fast_enough_velocity_untouched` | a floor that always fires |
| `floor_leaves_velocity_exactly_at_the_floor` | the strict `<` at the boundary |
| `floor_leaves_a_motionless_bird_at_rest` | rescaling a zero vector → NaN. Measured: speed is exactly zero **iff** a bird has no neighbour in any radius *and* started at rest, which never happens at ten birds or more |
| `a_floor_of_zero_disables_it` | `min_speed = 0` must reproduce the pre-floor behaviour |
| `floor_equal_to_cap_gives_constant_speed` | the self-propelled-particle setup, where every bird flies at exactly one speed |
| `physics_is_immutable` | physics changing halfway through a rollout |
| `physics_rejects_a_radius_at_or_beyond_half_the_world` | a radius past `bounds/2`, where minimum-image stops distinguishing near from far and the local rules quietly go global |
| `physics_allows_a_floor_of_zero_and_a_floor_equal_to_the_cap` | over-strict validation: both endpoints are legal and both are useful |
| `physics_rejects_a_floor_above_the_cap` | a clamp with no satisfiable range |
| `physics_rejects_nonsense_scales` | negative or zero `bounds`, `dt`, `max_speed` |
| `the_test_baseline_tracks_the_shipped_config` | `world.py` restating a constant as a literal instead of deriving it from `Physics()`. The two files had already drifted once — alignment_radius 35.0 against 30.0. Checks the module-level names too, not just the config objects: the first version of this test checked only the objects and a mutation reintroducing the literal passed all 81 tests |

## test_separation.py

Push away from close neighbours, `1/d`, capped.

| test | catches |
|---|---|
| `force_points_away_from_neighbour` | the sign |
| `force_points_away_from_2nd_bird` | the sign, from the mirror bird |
| `isolated_bird_feels_nothing` | a force from out of range; NaN on an empty neighbourhood |
| `neighbour_across_the_seam_still_counts` | a neighbour 4 units away treated as 96 |
| `closer_neighbour_pushes_harder` | direction not normalised before scaling by `1/d` — every force comes out magnitude 1 |
| `symmetric_neighbours_cancel` | accumulation and sign errors together |
| `forces_are_averaged_not_summed` | the averaging convention the reconstruction test has to agree with |
| `single_bird_world` | the degenerate case a refactor forgets |
| `force_cap_limits_magnitude_without_rotating_it` | clamping `dist` instead of the magnitude — `offset/dist` stops being a unit vector, so the "cap of 1.0" yields 0.5 and repulsion *grows* with distance |
| `force_cap_leaves_distant_neighbours_alone` | a cap quietly rescaling every label in the dataset |

## test_alignment.py

Match the mean heading of nearby neighbours.

| test | catches |
|---|---|
| `alignment_steers_toward_the_neighbour_mean` | the basic value |
| `alignment_switches_off_when_the_flock_already_matches` | returning the raw mean instead of the difference — an aligned flock would accelerate into the cap forever |
| `alignment_ignores_distance_inside_the_radius` | separation's `1/d` dragged along by copy-paste |
| `alignment_selects_neighbours_by_position_not_velocity` | **a real bug this caught**: `enumerate(velocities)` made the radius check measure position-to-velocity distance, ~70 units, so every alignment force in every flock was exactly `[0,0]` and nothing raised |
| `alignment_neighbour_across_the_seam_counts` | the seam |
| `alignment_isolated_bird_feels_nothing` | returning `−my_v` for an empty neighbourhood, which would make emptiness a brake |

## test_cohesion.py

Steer toward the local centre of mass. The most dangerous rule to get wrong.

| test | catches |
|---|---|
| `cohesion_pulls_toward_a_lone_neighbour` | the sign — separation and cohesion differ only by the order of two arguments to `get_torus` |
| `cohesion_averages_offsets_not_positions` | **the famous trap**: the mean of `x=1` and `x=99` is `50`, the far side of the world. Averaging positions gives a force in the opposite direction, 100× too strong, entirely plausible-looking |
| `cohesion_neighbour_across_the_seam_counts` | the seam |
| `cohesion_switches_off_at_the_centre_of_mass` | a rule that never reaches equilibrium |
| `cohesion_is_small_just_off_the_centre` | normalising the mean to a unit vector — a bird 0.02 off centre would get a full-magnitude force, discontinuous, and its direction near a flock's centre is mostly noise |
| `cohesion_grows_linearly_with_displacement` | anything that isn't Hooke's law: normalisation, or an inherited `1/d` |
| `cohesion_magnitude_never_reaches_one` | the bound that makes a force cap unnecessary here |
| `cohesion_excludes_neighbours_outside_the_radius` | an out-of-range bird entering the average, and inflating the divisor |
| `cohesion_counts_a_coincident_neighbour` | the deliberate absence of a `0 < dist` guard |
| `cohesion_isolated_bird_feels_nothing_and_divides_by_nothing` | **a real bug this caught**: the zero-guard placed *after* `total / neighbor_count`, so an isolated bird computed `0/0` → `[nan, nan]` plus a RuntimeWarning before returning the right answer. Under `np.seterr(all='raise')` that is a hard crash mid-run |
| `cohesion_single_bird_world` | the degenerate case |

## test_step.py

One tick: the integrator, the constants, and the invariants the labels rest on.

| test | catches |
|---|---|
| `recorded_forces_reconstruct_the_velocity_update` | **the load-bearing test.** If the recorded forces cannot reproduce the velocity update the simulator performed, they are not labels for this simulation. Guards its own premise by asserting all three rules actually fire |
| `reconstruction_holds_when_the_speed_cap_fires` | the same invariant where it stops being a plain sum: `clamp(v + Σf·dt) == v'` |
| `recorded_forces_match_the_rules_applied_to_the_input_state` | **the strongest label assertion here.** Every bird's every force must equal its own rule at its own radius times its own weight, on the state passed *in* — which pins the labels to tick *t* rather than *t+1*. Nothing else can: forces computed from the updated positions still satisfy the reconstruction invariant |
| `the_reconstruction_invariant_is_exact_in_float64` | rounding introduced by a reordering. Measured error is exactly zero, so this asserts equality, not closeness |
| `the_reconstruction_invariant_survives_float32_storage` | precision lost somewhere other than the cast. Bounds the round-trip error at `1e-6`; measured `2.9e-07`, about 2.4× float32 epsilon |
| `step_accepts_integer_input_without_truncating` | `np.zeros_like(positions)` inheriting int64 — every bird snaps to a 1×1 lattice every tick and the flock is wrong in a way that looks like a rendering choice |
| `step_always_computes_in_float64` | physics silently inheriting the caller's precision |
| `positions_stay_inside_the_world` | the upper seam |
| `positions_wrap_the_lower_seam` | the lower seam — Python's `%` returns non-negative for a positive modulus, but C's `fmod` would leave `−0.4` here |
| `step_reads_the_separation_weight` | **a real bug this caught**: deleting `* w_sep` passed all 20 tests that existed at the time. Unequal weights are what spread the three rules' acquisition times apart, so an ignored weight deletes the experiment's independent variable |
| `step_reads_the_separation_force_cap` | **a real bug this caught**: `step` accepted `sep_force_cap` and never forwarded it, so the callee's default set the physics and the recorded config was a lie |
| `step_reads_the_alignment_weight` | the same hole for `w_ali` |
| `step_reads_the_alignment_radius` | the same hole for `alignment_radius` |
| `step_reads_the_cohesion_weight` | the same hole for `w_coh` |
| `step_reads_the_cohesion_radius` | the same hole for `cohesion_radius` |
| `step_reads_the_min_speed` | the same hole for `min_speed` |
| `step_reads_the_dt` | **a real gap a mutation audit found**: every test ran at `dt=0.1`, so a literal was indistinguishable from a read |
| `step_reads_the_bounds` | **the same gap for `bounds`**: every test ran in a 100-wide world |
| `step_does_not_mutate_its_inputs` | double-buffering: `positions %= bounds` done in place would compute some birds' forces against a half-updated world, and the reconstruction test would keep passing |
| `step_records_all_three_rules_separately` | two keys holding the same array — probes trained to distinguish rules the simulator never separated |

## test_rollout.py

Hundreds of ticks. A different class of bug lives here.

| test | catches |
|---|---|
| `no_nan_ever_appears` | a NaN born on tick 300 that propagates silently to the end of the run |
| `positions_stay_inside_the_world_for_every_tick` | slow drift out of `[0, bounds)`, after which minimum-image is wrong for everything |
| `speed_never_exceeds_the_cap` | the cap failing under accumulated force |
| `speed_stays_between_the_floor_and_the_cap_on_every_tick` | either bound failing at any point in 500 ticks — and a bird stalling to exactly zero, which the floor cannot rescue |
| `the_reconstruction_invariant_holds_on_every_tick` | the load-bearing invariant, for every bird on every tick of a real flock |
| `the_reconstruction_invariant_holds_when_the_cap_fires_constantly` | the capped regime specifically — measured, a settled flock reaches its cap on 1 tick in 200, so an ordinary rollout barely tests it |
| `the_reconstruction_invariant_holds_with_the_speed_floor` | the same invariant under the other clamp |
| `a_cruising_flock_keeps_moving` | the reason the floor exists. Without it, alignment steers each bird toward its neighbours' mean *vector*, the mean of scattered vectors is short, and the flock organises its headings and brakes to a crawl in the same motion |
| `the_flock_settles_rather_than_being_pinned_at_the_cap` | an integrator driving the flock instead of letting it settle |
| `the_cohesion_spring_does_not_gain_energy` | plain forward Euler. Cohesion is Hooke's law, and forward Euler grows an oscillator's amplitude every period. `step` advances position with the **new** velocity — semi-implicit (symplectic) Euler — and that one line is what keeps the flock from shaking itself apart |
| `a_rollout_is_reproducible` | non-determinism, which would make the dataset unregenerable from its own metadata |
| `two_birds_beyond_every_radius_just_drift` | a rule firing when nothing is in range |
| `the_shipped_defaults_produce_a_sane_rollout` | `Physics()` with no arguments — what a careless caller gets, and what every other test overrides |
