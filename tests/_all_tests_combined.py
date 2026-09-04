# ==========================================================================
# world.py
# ==========================================================================
# """The world every test in this suite runs in.
#
# Every constant here derives from boids.py's Physics() rather than being
# restated, so the test suite and the simulator cannot disagree about the size of
# the world or how fast a bird may fly. There is one declaration, in boids.py.
# """
#
# from dataclasses import replace
#
# import numpy as np
#
# from boids import Physics
#
# # PHYSICS_CRUISING is the shipped configuration, taken from boids.py rather
# # than restated here. Everything below derives from it.
# #
# # The previous version declared BOUNDS, ALIGNMENT_RADIUS and the rest as
# # literals and fed them back into Physics(...), which is a second source of
# # truth for the same quantities - and they had already drifted apart, with
# # alignment_radius 35.0 here against 30.0 in boids.py. Deriving makes that
# # class of divergence impossible rather than merely fixing today's instance.
# PHYSICS_CRUISING = Physics()
#
# # The same world with the speed floor switched off, so that every test written
# # before the floor existed still means exactly what it meant then. Tests that
# # vary one constant use dataclasses.replace(PHYSICS, thing=...).
# PHYSICS = replace(PHYSICS_CRUISING, min_speed=0.0)
#
# BOUNDS = PHYSICS.bounds
# SEPARATION_RADIUS = PHYSICS.separation_radius
# ALIGNMENT_RADIUS = PHYSICS.alignment_radius
# COHESION_RADIUS = PHYSICS.cohesion_radius
# MAX_SPEED = PHYSICS.max_speed
# MIN_SPEED = PHYSICS_CRUISING.min_speed
# DT = PHYSICS.dt
#
#
# def random_flock(seed, n_birds):
#     """A deterministic starting world. Same seed, same flock, always."""
#     rng = np.random.default_rng(seed)
#     positions = rng.uniform(0.0, BOUNDS, size=(n_birds, 2))
#     velocities = rng.uniform(-MAX_SPEED, MAX_SPEED, size=(n_birds, 2))
#     return positions, velocities

# ==========================================================================
# primitives/test_geometry.py
# ==========================================================================
# """The primitives every rule is built on: torus distance, the speed cap,
# and the Physics config object's own validation."""
#
# from dataclasses import FrozenInstanceError, asdict
#
# import numpy as np
# import pytest
#
# from boids import Physics, get_torus, velocity_capper
# from world import (ALIGNMENT_RADIUS, BOUNDS, COHESION_RADIUS, DT, MAX_SPEED,
#                    MIN_SPEED, PHYSICS, PHYSICS_CRUISING, SEPARATION_RADIUS)
#
#
# def test_short_delta_passes_through_untouched():
#     """Birds close together: the direct route is already the shortest one,
#     so nothing should wrap."""
#     a = np.array([10.0, 10.0])
#     b = np.array([15.0, 10.0])
#
#     delta = get_torus(a, b, BOUNDS)
#
#     # x: 10 - 15 = -5, y: 0. Both well inside half a screen (50).
#     np.testing.assert_allclose(delta, [-5.0, 0.0])
#
# def test_wraps_positive_delta():
#     """A raw delta of +90 on a 100-wide torus is really -10: the short way
#     round runs in the opposite direction."""
#     a = np.array([95.0, 50.0])
#     b = np.array([5.0, 50.0])
#
#     delta = get_torus(a, b, BOUNDS)
#
#     np.testing.assert_allclose(delta, [-10.0, 0.0])
#
# def test_wraps_negative_delta():
#     """The mirror of the case above. An implementation that only corrects one
#     sign gets this wrong, and gets it wrong silently."""
#     a = np.array([5.0, 50.0])
#     b = np.array([95.0, 50.0])
#
#     delta = get_torus(a, b, BOUNDS)
#
#     np.testing.assert_allclose(delta, [10.0, 0.0])
#
# def test_wrap_is_decided_per_axis():
#     """x is more than half a screen apart and must wrap; y is not and must not.
#     A single `if` on the whole vector cannot express two different decisions."""
#     a = np.array([95.0, 15.0])
#     b = np.array([5.0, 5.0])
#
#     delta = get_torus(a, b, BOUNDS)
#
#     # x: raw 90 -> short way round is 10 the other direction -> -10
#     # y: raw 10 -> already shortest -> unchanged
#     np.testing.assert_allclose(delta, [-10.0, 10.0])
#
# def test_handles_position_wound_past_the_bounds():
#     """A position outside [0, bounds) still resolves correctly.
#
#     Positions should be kept wrapped by the integrator, but the modulo form
#     does not depend on that invariant holding, and this pins that guarantee
#     so a future "simplification" cannot quietly remove it.
#     """
#     a = np.array([230.0, 0.0])
#     b = np.array([0.0, 0.0])
#
#     delta = get_torus(a, b, BOUNDS)
#
#     # 230 is 2.3 screens out; 230 mod 100 = 30, which is under half a screen.
#     np.testing.assert_allclose(delta, [30.0, 0.0])
#
# def test_capper_leaves_slow_velocity_untouched():
#     """Under the cap the velocity must come back completely unchanged."""
#     v = np.array([1.0, 2.0])  # speed ~2.24, well under 5
#
#     np.testing.assert_array_equal(velocity_capper(v, MAX_SPEED), v)
#
# def test_capper_shortens_the_arrow_without_turning_it():
#     """Capping limits magnitude only - heading must survive untouched.
#
#     An implementation that returns `max_speed` itself passes a magnitude
#     check by accident while pointing the bird somewhere it was never going,
#     so the direction assertion is the one doing the work here.
#     """
#     v = np.array([6.0, 8.0])  # speed exactly 10
#     out = velocity_capper(v, MAX_SPEED)
#
#     assert np.linalg.norm(out) == pytest.approx(MAX_SPEED)
#     np.testing.assert_allclose(out / np.linalg.norm(out), v / np.linalg.norm(v))
#     np.testing.assert_allclose(out, [3.0, 4.0])
#
# def test_capper_leaves_velocity_exactly_at_the_cap():
#     """The boundary case: speed == max_speed must not be rescaled."""
#     v = np.array([3.0, 4.0])  # speed exactly 5
#
#     np.testing.assert_allclose(velocity_capper(v, MAX_SPEED), v)
#
# def test_capper_handles_a_motionless_bird():
#     """A stationary bird has speed 0. Normalising it would divide by zero and
#     produce NaN without raising, so the branch must not be taken at all."""
#     out = velocity_capper(np.array([0.0, 0.0]), MAX_SPEED)
#
#     assert not np.isnan(out).any(), "NaN velocity would spread to every later tick"
#     np.testing.assert_array_equal(out, [0.0, 0.0])
#
# def test_physics_is_immutable():
#     """A rollout must not be able to change the physics halfway through."""
#     physics = Physics()
#
#     with pytest.raises(FrozenInstanceError):
#         physics.dt = 0.2
#
# def test_physics_rejects_a_radius_at_or_beyond_half_the_world():
#     """The minimum-image convention returns at most bounds/2 along each axis.
#
#     A radius beyond that makes every bird a neighbour along that axis, so the
#     rule stops being local while still producing entirely plausible numbers.
#     """
#     with pytest.raises(ValueError, match="alignment_radius"):
#         Physics(bounds=100.0, alignment_radius=50.0)
#
#     with pytest.raises(ValueError, match="separation_radius"):
#         Physics(bounds=100.0, separation_radius=60.0)
#
# def test_physics_rejects_nonsense_scales():
#     with pytest.raises(ValueError):
#         Physics(bounds=-1.0)
#     with pytest.raises(ValueError):
#         Physics(dt=0.0)
#     with pytest.raises(ValueError):
#         Physics(max_speed=-5.0)
#
#
# # --- velocity_capper: the speed floor ----------------------------------------
#
# def test_floor_raises_a_slow_velocity_without_turning_it():
#     """The mirror of the cap: rescale the magnitude, keep the heading.
#
#     Speed 1.0 against a floor of 2.0 must double the vector, not rotate it.
#     """
#     out = velocity_capper(np.array([0.6, 0.8]), MAX_SPEED, MIN_SPEED)
#
#     np.testing.assert_allclose(out, [1.2, 1.6])
#     assert np.linalg.norm(out) == pytest.approx(MIN_SPEED)
#
#
# def test_floor_leaves_a_fast_enough_velocity_untouched():
#     velocity = np.array([3.0, 0.0])
#
#     np.testing.assert_array_equal(
#         velocity_capper(velocity, MAX_SPEED, MIN_SPEED), velocity)
#
#
# def test_floor_leaves_velocity_exactly_at_the_floor():
#     """`0 < speed < min_speed` is strict at both ends, deliberately."""
#     velocity = np.array([MIN_SPEED, 0.0])
#
#     np.testing.assert_array_equal(
#         velocity_capper(velocity, MAX_SPEED, MIN_SPEED), velocity)
#
#
# def test_floor_leaves_a_motionless_bird_at_rest():
#     """A zero vector has no direction to rescale, so it stays put.
#
#     Measured over 60,000 bird-ticks: speed is exactly zero if and only if a
#     bird has no neighbour in any radius AND started at rest, which never
#     happens at ten birds or more. This branch is a guard against dividing by
#     zero, not a behaviour anything relies on.
#     """
#     out = velocity_capper(np.array([0.0, 0.0]), MAX_SPEED, MIN_SPEED)
#
#     assert not np.isnan(out).any(), "a zero vector must not be rescaled"
#     np.testing.assert_array_equal(out, [0.0, 0.0])
#
#
# def test_a_floor_of_zero_disables_it():
#     """min_speed = 0.0 must reproduce the behaviour from before the floor."""
#     velocity = np.array([0.001, 0.0])
#
#     np.testing.assert_array_equal(
#         velocity_capper(velocity, MAX_SPEED, 0.0), velocity)
#
#
# def test_floor_equal_to_cap_gives_constant_speed():
#     """The self-propelled-particle setup: every bird flies at exactly one speed."""
#     slow = velocity_capper(np.array([0.5, 0.0]), 2.0, 2.0)
#     fast = velocity_capper(np.array([9.0, 0.0]), 2.0, 2.0)
#
#     np.testing.assert_allclose(slow, [2.0, 0.0])
#     np.testing.assert_allclose(fast, [2.0, 0.0])
#
#
# def test_physics_allows_a_floor_of_zero_and_a_floor_equal_to_the_cap():
#     """Both ends of the range are legal and both are useful.
#
#     min_speed = 0 means no floor, which is what ablations and any test that
#     needs the floor out of the way will ask for. min_speed = max_speed is the
#     constant-speed flocking model. Neither is an error.
#     """
#     assert Physics(min_speed=0.0).min_speed == 0.0
#     assert Physics(min_speed=5.0, max_speed=5.0).min_speed == 5.0
#
#
# def test_physics_rejects_a_floor_above_the_cap():
#     with pytest.raises(ValueError):
#         Physics(min_speed=6.0, max_speed=5.0)
#     with pytest.raises(ValueError):
#         Physics(min_speed=-1.0)
#
#
# def test_the_test_baseline_tracks_the_shipped_config():
#     """world.py must derive from boids.py, not restate it.
#
#     These two files once declared the same constants independently and drifted:
#     alignment_radius was 35.0 in world.py against 30.0 in boids.py, and during a
#     planning session a measurement taken with one was reported alongside the
#     other. Deriving makes that impossible, and this asserts the derivation
#     rather than trusting it - if anyone reintroduces a literal, the two configs
#     can differ again and this is what notices.
#
#     PHYSICS_CRUISING is the shipped config verbatim. PHYSICS is the same world
#     with the speed floor switched off, and that is the *only* difference.
#     """
#     assert PHYSICS_CRUISING == Physics(), (
#         "PHYSICS_CRUISING must be boids.py's Physics() unchanged")
#
#     shipped, baseline = asdict(Physics()), asdict(PHYSICS)
#     differing = {k for k in shipped if shipped[k] != baseline[k]}
#     assert differing == {"min_speed"}, (
#         f"the test baseline differs from the shipped config in {differing}, "
#         "but min_speed is the only difference it is allowed to have"
#     )
#
#     # And the module-level names must be the config's own values, not copies.
#     # Checking only the Physics objects leaves the original bug available: a
#     # literal reintroduced here would disagree with PHYSICS.alignment_radius
#     # while both objects stayed correct. A mutation test found exactly that
#     # hole in the first version of this test.
#     assert (BOUNDS, SEPARATION_RADIUS, ALIGNMENT_RADIUS,
#             COHESION_RADIUS, MAX_SPEED, DT) == (
#         PHYSICS.bounds, PHYSICS.separation_radius, PHYSICS.alignment_radius,
#         PHYSICS.cohesion_radius, PHYSICS.max_speed, PHYSICS.dt), (
#         "a world.py constant has been restated as a literal instead of derived")
#     assert MIN_SPEED == PHYSICS_CRUISING.min_speed

# ==========================================================================
# forces/test_separation.py
# ==========================================================================
# """calculate_separation: push away from close neighbours, 1/d, capped."""
#
# import numpy as np
# import pytest
#
# from boids import calculate_separation
# from world import BOUNDS, SEPARATION_RADIUS
#
#
# def test_force_points_away_from_neighbour():
#     """Bird 0 has a neighbour 5 units away in +x, so it must be pushed in -x.
#
#     A flipped sign here turns separation into attraction. Nothing raises, the
#     flock still looks like a flock, and every f_sep label in the dataset is
#     wrong - so this assertion is the one holding the convention in place.
#     """
#     positions = np.array([[10.0, 10.0],
#                           [15.0, 10.0]])
#
#     force = calculate_separation(0, positions, SEPARATION_RADIUS, BOUNDS)
#
#     assert force[0] < 0, "bird 0 must flee in -x, away from the bird at +x"
#     assert force[1] == pytest.approx(0.0), "no y neighbour, so no y force"
#     # magnitude is 1/d, and the birds are d = 5 apart
#     assert np.linalg.norm(force) == pytest.approx(0.2)
#
# def test_force_points_away_from_2nd_bird():
#     """The mirror of the test above: same world, other bird.
#
#     Not redundant. A bug that reads `positions[0]` instead of
#     `positions[boid_id]` passes the test above and fails this one.
#     """
#     positions = np.array([[10.0, 10.0],
#                           [15.0, 10.0]])
#
#     force = calculate_separation(1, positions, SEPARATION_RADIUS, BOUNDS)
#
#     assert force[0] > 0, "bird 1 must flee in +x, away from the bird at -x"
#     assert force[1] == pytest.approx(0.0), "no y neighbour, so no y force"
#     # magnitude is 1/d, and the birds are d = 5 apart
#     assert np.linalg.norm(force) == pytest.approx(0.2)
#
# def test_isolated_bird_feels_nothing():
#     """Nearest neighbour is ~42 away against a radius of 20, so nothing counts.
#
#     Must return exactly zero - and must not return NaN, which is what a
#     divide-by-zero produces here without raising anything.
#     """
#     positions = np.array([[10.0, 10.0],
#                           [80.0, 80.0]])
#
#     force = calculate_separation(0, positions, SEPARATION_RADIUS, BOUNDS)
#
#     assert not np.isnan(force).any(), "NaN would silently poison every label"
#     np.testing.assert_array_equal(force, [0.0, 0.0])
#
# def test_neighbour_across_the_seam_still_counts():
#     """Birds at x=2 and x=98 on a 100-wide torus are 4 apart, not 96.
#
#     An implementation that lost the wrap would see nothing inside the radius
#     and return zero here. This is the test that proves the world is a torus.
#     """
#     positions = np.array([[2.0, 50.0],
#                           [98.0, 50.0]])
#
#     force = calculate_separation(0, positions, SEPARATION_RADIUS, BOUNDS)
#
#     # d = 4, so magnitude 1/4; bird 0 flees in +x, through the seam.
#     np.testing.assert_allclose(force, [0.25, 0.0])
#
# def test_closer_neighbour_pushes_harder():
#     """Magnitude must fall off as 1/d.
#
#     This is the test that catches the cancellation bug: if the direction
#     vector is not normalised before scaling by 1/d, every force comes out
#     with magnitude exactly 1 regardless of distance, and both worlds below
#     would return the same answer.
#     """
#     far = np.array([[10.0, 50.0],
#                     [18.0, 50.0]])   # d = 8
#     near = np.array([[10.0, 50.0],
#                      [14.0, 50.0]])  # d = 4
#
#     force_far = calculate_separation(0, far, SEPARATION_RADIUS, BOUNDS)
#     force_near = calculate_separation(0, near, SEPARATION_RADIUS, BOUNDS)
#
#     assert np.linalg.norm(force_far) == pytest.approx(1 / 8)
#     assert np.linalg.norm(force_near) == pytest.approx(1 / 4)
#     assert np.linalg.norm(force_near) > np.linalg.norm(force_far)
#
# def test_symmetric_neighbours_cancel():
#     """Equal neighbours either side leave the middle bird with no net force.
#
#     Catches sign errors and accumulation errors at once. Note it would also
#     pass for a function that always returned zero - which is exactly why it
#     is an addition to the suite rather than a replacement for it.
#     """
#     positions = np.array([[5.0, 50.0],
#                           [10.0, 50.0],
#                           [15.0, 50.0]])
#
#     force = calculate_separation(1, positions, SEPARATION_RADIUS, BOUNDS)
#
#     np.testing.assert_allclose(force, [0.0, 0.0], atol=1e-12)
#
# def test_forces_are_averaged_not_summed():
#     """Two neighbours on the same side, at d = 5 and d = 10.
#
#     Individual magnitudes are 1/5 and 1/10, both pointing -x, so the sum is
#     -0.3 and the average is -0.15. This pins the averaging convention, which
#     is a choice - and one the reconstruction test will have to agree with.
#     """
#     positions = np.array([[10.0, 50.0],
#                           [15.0, 50.0],
#                           [20.0, 50.0]])
#
#     force = calculate_separation(0, positions, SEPARATION_RADIUS, BOUNDS)
#
#     np.testing.assert_allclose(force, [-0.15, 0.0])
#
# def test_single_bird_world():
#     """A world with one bird. No neighbours to skip, nothing to divide by.
#
#     The degenerate case a refactor is most likely to forget, and it must
#     return zero rather than raising or producing NaN.
#     """
#     positions = np.array([[10.0, 10.0]])
#
#     force = calculate_separation(0, positions, SEPARATION_RADIUS, BOUNDS)
#
#     assert not np.isnan(force).any()
#     np.testing.assert_array_equal(force, [0.0, 0.0])
#
# def test_force_cap_limits_magnitude_without_rotating_it():
#     """Two birds 0.5 apart. Uncapped the 1/d law gives magnitude 2.0.
#
#     The cap must bound the magnitude and leave the heading alone - exactly
#     what `velocity_capper` does for speed. The failure this exists to catch
#     is clamping `dist` itself instead of the magnitude expression: that makes
#     `offset / dist` a half-length vector rather than a unit one, so the force
#     comes out at 0.5 instead of the capped 1.0, and inside the cap radius the
#     repulsion would grow with distance instead of shrinking. Nothing raises.
#     """
#     positions = np.array([[10.0, 50.0],
#                           [10.5, 50.0]])
#
#     uncapped = calculate_separation(
#         0, positions, SEPARATION_RADIUS, BOUNDS, sep_force_cap=10.0
#     )
#     capped = calculate_separation(
#         0, positions, SEPARATION_RADIUS, BOUNDS, sep_force_cap=1.0
#     )
#
#     # Guard the premise: if the cap never bites, this test proves nothing.
#     assert np.linalg.norm(uncapped) == pytest.approx(2.0)
#
#     np.testing.assert_allclose(capped, [-1.0, 0.0])
#     assert np.linalg.norm(capped) == pytest.approx(1.0), (
#         "magnitude should be the cap exactly - 0.5 means `dist` was clamped "
#         "and `direction` is no longer a unit vector"
#     )
#
# def test_force_cap_leaves_distant_neighbours_alone():
#     """A neighbour at d = 5 gives magnitude 0.2, far under the cap.
#
#     The cap must be inert everywhere except very close range, or it is
#     quietly rescaling every label in the dataset.
#     """
#     positions = np.array([[10.0, 50.0],
#                           [15.0, 50.0]])
#
#     force = calculate_separation(
#         0, positions, SEPARATION_RADIUS, BOUNDS, sep_force_cap=1.0
#     )
#
#     np.testing.assert_allclose(force, [-0.2, 0.0])

# ==========================================================================
# forces/test_alignment.py
# ==========================================================================
# """calculate_alignment: match the mean heading of nearby neighbours."""
#
# import numpy as np
#
# from boids import calculate_alignment
# from world import ALIGNMENT_RADIUS, BOUNDS
#
#
# def test_alignment_steers_toward_the_neighbour_mean():
#     """Three birds in a tight cluster, all headed differently.
#
#     Bird 0 sees velocities [0,1] and [-1,0], mean [-0.5, 0.5], and is itself
#     doing [1,0]. The steering force is the difference: [-1.5, 0.5].
#     """
#     positions = np.array([[50.0, 50.0],
#                           [52.0, 50.0],
#                           [54.0, 50.0]])
#     velocities = np.array([[1.0, 0.0],
#                            [0.0, 1.0],
#                            [-1.0, 0.0]])
#
#     force = calculate_alignment(0, positions, velocities, ALIGNMENT_RADIUS, BOUNDS)
#
#     np.testing.assert_allclose(force, [-1.5, 0.5])
#
# def test_alignment_switches_off_when_the_flock_already_matches():
#     """The trap this rule exists around.
#
#     Returning the raw neighbour mean instead of the difference from your own
#     velocity gives an already-aligned flock a permanent force of [3,0] - it
#     would accelerate forever into the speed cap and never settle. A matched
#     flock must feel nothing.
#     """
#     positions = np.array([[50.0, 50.0],
#                           [52.0, 50.0],
#                           [54.0, 50.0]])
#     velocities = np.array([[3.0, 0.0],
#                            [3.0, 0.0],
#                            [3.0, 0.0]])
#
#     force = calculate_alignment(0, positions, velocities, ALIGNMENT_RADIUS, BOUNDS)
#
#     np.testing.assert_allclose(force, [0.0, 0.0], atol=1e-12)
#
# def test_alignment_ignores_distance_inside_the_radius():
#     """No 1/d falloff - that is separation's law, not this one.
#
#     A neighbour at d = 5 and the same neighbour at d = 25 must produce an
#     identical force. This is what catches alignment written by copy-pasting
#     calculate_separation and forgetting to strip the distance weighting.
#
#     The far distance is 25 rather than 30 because ALIGNMENT_RADIUS now derives
#     from boids.py and is 30.0; the comparison is a strict <, so a neighbour at
#     exactly 30 would fall outside and this would be testing nothing.
#     """
#     velocities = np.array([[1.0, 0.0],
#                            [0.0, 1.0]])
#     near = np.array([[50.0, 50.0], [55.0, 50.0]])   # d = 5
#     far = np.array([[50.0, 50.0], [75.0, 50.0]])    # d = 25, inside 30
#
#     force_near = calculate_alignment(0, near, velocities, ALIGNMENT_RADIUS, BOUNDS)
#     force_far = calculate_alignment(0, far, velocities, ALIGNMENT_RADIUS, BOUNDS)
#
#     np.testing.assert_allclose(force_near, [-1.0, 1.0])
#     np.testing.assert_allclose(force_far, force_near)
#
# def test_alignment_selects_neighbours_by_position_not_velocity():
#     """Regression test for a real bug: `enumerate(velocities)`.
#
#     Iterating the wrong array made `neighbor_pos` a velocity, so the radius
#     check measured the distance from a position at [50,50] to a point at
#     [0,1] - about 70 units, outside every sane radius. Every bird's alignment
#     force came out as exactly [0,0], in every flock, and nothing raised: the
#     summation line indexed `velocities[i]` correctly, so the function was
#     averaging a legitimate set of velocities. Just the empty one.
#     """
#     positions = np.array([[50.0, 50.0],
#                           [52.0, 50.0]])       # 2 units apart, plainly neighbours
#     velocities = np.array([[1.0, 0.0],
#                            [0.0, 1.0]])        # 70 units apart as coordinates
#
#     force = calculate_alignment(0, positions, velocities, ALIGNMENT_RADIUS, BOUNDS)
#
#     np.testing.assert_allclose(force, [-1.0, 1.0])
#     assert np.linalg.norm(force) > 0, (
#         "zero force for adjacent birds means neighbours were selected by "
#         "velocity rather than by position"
#     )
#
# def test_alignment_neighbour_across_the_seam_counts():
#     """x = 2 and x = 98 are 4 units apart on a torus, not 96."""
#     positions = np.array([[2.0, 50.0],
#                           [98.0, 50.0]])
#     velocities = np.array([[1.0, 0.0],
#                            [0.0, 2.0]])
#
#     force = calculate_alignment(0, positions, velocities, ALIGNMENT_RADIUS, BOUNDS)
#
#     np.testing.assert_allclose(force, [-1.0, 2.0])
#
# def test_alignment_isolated_bird_feels_nothing():
#     """No neighbours means no force - not minus its own velocity.
#
#     A lone bird coasts. Returning `-my_v` for an empty neighbourhood would
#     make emptiness act as a brake, which is a different rule than the one
#     being modelled.
#     """
#     positions = np.array([[10.0, 10.0],
#                           [50.0, 50.0]])   # torus distance 56.6, outside 35
#     velocities = np.array([[4.0, 4.0],
#                            [0.0, -1.0]])
#
#     force = calculate_alignment(0, positions, velocities, ALIGNMENT_RADIUS, BOUNDS)
#
#     assert not np.isnan(force).any()
#     np.testing.assert_array_equal(force, [0.0, 0.0])

# ==========================================================================
# forces/test_cohesion.py
# ==========================================================================
# """calculate_cohesion: steer toward the local centre of mass.
#
# Two things make this the most dangerous of the three rules to get wrong.
#
# The centre of mass of a set of positions is meaningless on a torus - the mean
# of x=1 and x=99 is 50, the far side of the world - so the rule must average
# *offsets*, and several tests here exist only to prove that it does.
#
# And the force is a spring: magnitude proportional to displacement from the
# centre, rather than the 1/d law of separation or the flat average of
# alignment. Normalising it to a unit vector would be the natural thing to write
# and would silently destroy the label, so that is pinned too.
# """
#
# import warnings
#
# import numpy as np
# import pytest
#
# from boids import calculate_cohesion
# from world import BOUNDS, COHESION_RADIUS
#
#
# def test_cohesion_pulls_toward_a_lone_neighbour():
#     """The sign test.
#
#     Separation and cohesion differ by the order of two arguments to get_torus
#     and nothing else. Swapping them inverts the physics, raises nothing, and
#     produces a force vector of exactly the right magnitude.
#     """
#     positions = np.array([[50.0, 50.0],
#                           [60.0, 50.0]])   # neighbour 10 units in +x
#
#     force = calculate_cohesion(0, positions, COHESION_RADIUS, BOUNDS)
#
#     np.testing.assert_allclose(force, [0.25, 0.0])   # 10 / 40
#     assert force[0] > 0, (
#         "cohesion must pull toward the neighbour; a negative x here means the "
#         "arguments to get_torus are the wrong way round and this is separation"
#     )
#
#
# def test_cohesion_averages_offsets_not_positions():
#     """The trap this rule is famous for.
#
#     Bird 0 sits at x=0 with neighbours at x=2 and x=99 - which on a torus are
#     2 units to its right and 1 unit to its left. The true mean offset is
#     [0.5, 0], a gentle pull in +x.
#
#     Average the raw positions instead and you get (2 + 99) / 2 = 50.5, i.e. a
#     centre of mass on the far side of the world, and a force of about
#     [-1.24, 0]: the opposite direction, a hundred times too strong, and
#     entirely plausible-looking.
#     """
#     positions = np.array([[0.0, 50.0],
#                           [2.0, 50.0],
#                           [99.0, 50.0]])
#
#     force = calculate_cohesion(0, positions, COHESION_RADIUS, BOUNDS)
#
#     np.testing.assert_allclose(force, [0.0125, 0.0])
#     assert force[0] > 0, "a negative force here means positions were averaged"
#
#
# def test_cohesion_neighbour_across_the_seam_counts():
#     """x=2 and x=98 are 4 units apart, and the pull is leftward through 0."""
#     positions = np.array([[2.0, 50.0],
#                           [98.0, 50.0]])
#
#     force = calculate_cohesion(0, positions, COHESION_RADIUS, BOUNDS)
#
#     np.testing.assert_allclose(force, [-0.1, 0.0])   # -4 / 40
#
#
# def test_cohesion_switches_off_at_the_centre_of_mass():
#     """A bird already at the centre of its flock must feel nothing.
#
#     This is the property that normalising the mean to a unit vector destroys:
#     under normalisation a bird at the exact centre gets [0,0] but a bird a
#     thousandth of a unit away gets a full-magnitude force.
#     """
#     positions = np.array([[50.0, 50.0],
#                           [45.0, 50.0],
#                           [55.0, 50.0]])
#
#     force = calculate_cohesion(0, positions, COHESION_RADIUS, BOUNDS)
#
#     np.testing.assert_allclose(force, [0.0, 0.0], atol=1e-12)
#
#
# def test_cohesion_is_small_just_off_the_centre():
#     """The other half of the previous test, and the one with teeth.
#
#     Bird 0 is 0.02 units off its flock's centre of mass. The force must be
#     correspondingly tiny. An implementation that returns mean / norm(mean)
#     returns magnitude 1.0 here - two thousand times too large - while still
#     passing every test that only checks direction.
#     """
#     positions = np.array([[50.0, 50.0],
#                           [45.0, 50.0],
#                           [55.04, 50.0]])
#
#     force = calculate_cohesion(0, positions, COHESION_RADIUS, BOUNDS)
#
#     np.testing.assert_allclose(force, [0.0005, 0.0])
#     assert np.linalg.norm(force) < 0.001, (
#         "a magnitude near 1.0 means the mean offset was normalised, which "
#         "makes the force discontinuous at the centre of mass"
#     )
#
#
# def test_cohesion_grows_linearly_with_displacement():
#     """Hooke's law: magnitude proportional to distance from the centre.
#
#     Three worlds, one neighbour each, at 4, 20 and 39 units. Any rule that is
#     normalised, or that inherits separation's 1/d falloff, breaks this.
#     """
#     forces = [
#         calculate_cohesion(
#             0,
#             np.array([[50.0, 50.0], [50.0 + d, 50.0]]),
#             COHESION_RADIUS,
#             BOUNDS,
#         )
#         for d in (4.0, 20.0, 39.0)
#     ]
#     magnitudes = [np.linalg.norm(f) for f in forces]
#
#     assert magnitudes == pytest.approx([0.1, 0.5, 0.975])
#     assert magnitudes[0] < magnitudes[1] < magnitudes[2]
#
#
# def test_cohesion_magnitude_never_reaches_one():
#     """Bounded by construction, which is why this rule needs no force cap.
#
#     Every offset in the sum has norm below cohesion_radius, so their mean does
#     too, so mean / cohesion_radius is inside the unit disc. Thirty birds
#     scattered at random must all obey it.
#     """
#     rng = np.random.default_rng(0)
#     positions = rng.uniform(0.0, BOUNDS, size=(30, 2))
#
#     magnitudes = np.array([
#         np.linalg.norm(calculate_cohesion(i, positions, COHESION_RADIUS, BOUNDS))
#         for i in range(len(positions))
#     ])
#
#     assert magnitudes.max() < 1.0
#     # premise guard: a function returning zeros would satisfy the bound
#     assert magnitudes.max() > 0.0
#
#
# def test_cohesion_excludes_neighbours_outside_the_radius():
#     """A third bird 45 units away must not enter the average at all.
#
#     If it did, the answer would differ from the two-bird case below - and the
#     divisor would be 2 rather than 1, which is the quieter half of the bug.
#     """
#     with_far_bird = np.array([[50.0, 50.0],
#                               [60.0, 50.0],
#                               [95.0, 50.0]])   # 45 units away, outside 40
#
#     force = calculate_cohesion(0, with_far_bird, COHESION_RADIUS, BOUNDS)
#
#     np.testing.assert_allclose(force, [0.25, 0.0])
#
#
# def test_cohesion_counts_a_coincident_neighbour():
#     """Unlike separation, cohesion has no `0 < dist` guard, deliberately.
#
#     Nothing divides by dist here, so a bird sitting exactly on top of another
#     cannot produce a NaN. It contributes a zero offset and still counts toward
#     the divisor: mean of [0,0] and [10,0] is [5,0], not [10,0].
#     """
#     positions = np.array([[50.0, 50.0],
#                           [50.0, 50.0],
#                           [60.0, 50.0]])
#
#     force = calculate_cohesion(0, positions, COHESION_RADIUS, BOUNDS)
#
#     assert not np.isnan(force).any()
#     np.testing.assert_allclose(force, [0.125, 0.0])   # 5 / 40
#
#
# def test_cohesion_isolated_bird_feels_nothing_and_divides_by_nothing():
#     """No neighbours means no force - and no 0/0 on the way there.
#
#     numpy answers np.zeros(2) / 0 with [nan, nan] and a RuntimeWarning rather
#     than an exception, so a guard placed after the division still returns the
#     right answer while quietly emitting a warning per isolated bird per tick.
#     Under np.seterr(all='raise') - which the dataset generator should use - it
#     becomes a hard crash mid-run instead.
#     """
#     positions = np.array([[10.0, 10.0],
#                           [50.0, 50.0]])   # torus distance 56.6, outside 40
#
#     with warnings.catch_warnings():
#         warnings.simplefilter("error")   # any RuntimeWarning fails this test
#         force = calculate_cohesion(0, positions, COHESION_RADIUS, BOUNDS)
#
#     assert not np.isnan(force).any()
#     np.testing.assert_array_equal(force, [0.0, 0.0])
#
#
# def test_cohesion_single_bird_world():
#     """One bird, no neighbours to skip, nothing to divide by."""
#     positions = np.array([[10.0, 10.0]])
#
#     with warnings.catch_warnings():
#         warnings.simplefilter("error")
#         force = calculate_cohesion(0, positions, COHESION_RADIUS, BOUNDS)
#
#     assert not np.isnan(force).any()
#     np.testing.assert_array_equal(force, [0.0, 0.0])

# ==========================================================================
# simulator/test_step.py
# ==========================================================================
# """step(): the integrator, and the reconstruction invariant.
#
# The invariant is the load-bearing test in this project. Every probe and every
# ablation is trained against the numbers in the forces dict, so if those numbers
# cannot reproduce the velocity update the simulator actually performed, they are
# not labels for this simulation and every downstream result is fitted to
# something that never happened.
# """
#
# from dataclasses import replace
#
# import numpy as np
#
# from boids import (calculate_alignment, calculate_cohesion,
#                    calculate_separation, step, velocity_capper)
# from world import (BOUNDS, DT, MAX_SPEED, MIN_SPEED, PHYSICS,
#                    PHYSICS_CRUISING, SEPARATION_RADIUS, random_flock)
#
#
# def _reconstruct_velocity(velocities, forces, boid_id, physics):
#     """Re-derive one bird's new velocity from the RECORDED forces alone.
#
#     This is the whole point of the exercise: if the numbers stored in the
#     forces dict cannot reproduce the velocity update the simulator actually
#     performed, then those numbers are not labels for this simulation, and
#     every probe trained on them is fitting something that never happened.
#     """
#     total = (forces["separation"][boid_id]
#              + forces["alignment"][boid_id]
#              + forces["cohesion"][boid_id])
#     return velocity_capper(velocities[boid_id] + total * physics.dt,
#                            physics.max_speed, physics.min_speed)
#
# def test_recorded_forces_reconstruct_the_velocity_update():
#     """The uncapped case: recorded forces sum straight to the velocity change."""
#     positions = np.array([[10.0, 10.0],
#                           [15.0, 10.0],
#                           [80.0, 80.0]])
#     velocities = np.array([[1.0, 0.0],
#                            [-1.0, 0.0],
#                            [0.5, 0.5]])
#
#     _, new_velocities, forces = step(positions, velocities, PHYSICS)
#
#     # Guard the premise. The invariant holds trivially if a rule contributes
#     # nothing, so this world must actually exercise all three of them.
#     for rule in ("separation", "alignment", "cohesion"):
#         assert np.any(forces[rule] != 0.0), (
#             f"{rule} is all zeros here, so this world does not test it"
#         )
#
#     for i in range(len(positions)):
#         np.testing.assert_allclose(
#             _reconstruct_velocity(velocities, forces, i, PHYSICS),
#             new_velocities[i],
#             err_msg=f"recorded forces do not reproduce bird {i}'s velocity update",
#         )
#
# def test_reconstruction_holds_when_the_speed_cap_fires():
#     """The case that actually matters.
#
#     Once the cap engages, recorded forces no longer sum to the velocity
#     change - the relationship is `cap(v_old + sum(f) * dt) == v_new`, not a
#     plain sum. A reconstruction test that never trips the cap verifies only
#     the easy half and would pass against a simulator whose labels are wrong
#     in exactly the regime where birds are moving fastest.
#     """
#     physics = replace(PHYSICS, max_speed=0.5)  # every bird below exceeds it
#     positions = np.array([[10.0, 10.0],
#                           [15.0, 10.0],
#                           [80.0, 80.0]])
#     velocities = np.array([[1.0, 0.0],
#                            [-1.0, 0.0],
#                            [0.5, 0.5]])
#
#     _, new_velocities, forces = step(positions, velocities, physics)
#
#     # Guard the test's own premise: if the cap never fires this proves nothing.
#     uncapped = np.array([
#         velocities[i] + (forces["separation"][i]
#                          + forces["alignment"][i]
#                          + forces["cohesion"][i]) * physics.dt
#         for i in range(len(positions))
#     ])
#     assert np.any(np.linalg.norm(uncapped, axis=1) > physics.max_speed), (
#         "premise failed: the cap never fired, so this test verifies nothing"
#     )
#
#     for i in range(len(positions)):
#         np.testing.assert_allclose(
#             _reconstruct_velocity(velocities, forces, i, physics),
#             new_velocities[i],
#             err_msg=f"reconstruction broke for bird {i} once the cap fired",
#         )
#         assert np.linalg.norm(new_velocities[i]) <= physics.max_speed + 1e-12
#
# def test_positions_stay_inside_the_world():
#     """`pos %= bounds` every tick is the invariant get_torus depends on.
#
#     Both birds start within half a step of an edge and are moving straight at
#     it, so each must wrap round rather than leaving the box.
#     """
#     positions = np.array([[99.9, 50.0],
#                           [50.0, 99.9]])
#     velocities = np.array([[10.0, 0.0],
#                            [0.0, 10.0]])
#
#     new_positions, _, _ = step(positions, velocities, PHYSICS)
#
#     assert np.all(new_positions >= 0.0)
#     assert np.all(new_positions < BOUNDS)
#     # confirm they wrapped rather than simply not reaching the edge
#     assert new_positions[0][0] < 1.0, "bird 0 should have crossed the x seam"
#     assert new_positions[1][1] < 1.0, "bird 1 should have crossed the y seam"
#
# def test_step_reads_the_separation_weight():
#     """`w_sep` must scale the recorded force, and proportionally.
#
#     Deleting `* w_sep` from `step` passed all twenty of the tests above. The
#     weights are not decoration - NOTES.md makes unequal weights the mechanism
#     that spreads the three rules' acquisition times apart, so a silently
#     ignored weight does not crash anything, it just deletes the experiment's
#     independent variable.
#
#     This asserts the ratio rather than a hardcoded 1.5, so retuning the
#     weights later does not turn this into a false alarm.
#     """
#     positions = np.array([[10.0, 50.0],
#                           [15.0, 50.0]])
#     velocities = np.zeros((2, 2))
#
#     raw = calculate_separation(0, positions, SEPARATION_RADIUS, BOUNDS)
#
#     _, _, forces_1x = step(positions, velocities, replace(PHYSICS, w_sep=1.0))
#     _, _, forces_2x = step(positions, velocities, replace(PHYSICS, w_sep=2.0))
#
#     # weight 1.0 must leave the raw force untouched: it multiplies, not adds
#     np.testing.assert_allclose(forces_1x["separation"][0], raw)
#     # and doubling the weight must double the recorded force
#     np.testing.assert_allclose(forces_2x["separation"][0],
#                                2.0 * forces_1x["separation"][0])
#
# def test_step_reads_the_separation_force_cap():
#     """`sep_force_cap` must reach `calculate_separation`.
#
#     A parameter that `step` accepts and then drops is invisible: the callee's
#     default silently sets the physics, the top-level config that gets recorded
#     alongside the dataset says nothing about it, and the run is not
#     reproducible from its own metadata.
#     """
#     positions = np.array([[10.0, 50.0],
#                           [10.5, 50.0]])   # d = 0.5, uncapped magnitude 2.0
#     velocities = np.zeros((2, 2))
#
#     _, _, tight = step(positions, velocities,
#                        replace(PHYSICS, sep_force_cap=1.0, w_sep=1.0))
#     _, _, loose = step(positions, velocities,
#                        replace(PHYSICS, sep_force_cap=10.0, w_sep=1.0))
#
#     np.testing.assert_allclose(tight["separation"][0], [-1.0, 0.0])
#     np.testing.assert_allclose(loose["separation"][0], [-2.0, 0.0])
#
# def test_step_reads_the_alignment_weight():
#     positions = np.array([[50.0, 50.0],
#                           [52.0, 50.0]])
#     velocities = np.array([[1.0, 0.0],
#                            [0.0, 1.0]])
#
#     _, _, forces_1x = step(positions, velocities, replace(PHYSICS, w_ali=1.0))
#     _, _, forces_2x = step(positions, velocities, replace(PHYSICS, w_ali=2.0))
#
#     np.testing.assert_allclose(forces_1x["alignment"][0], [-1.0, 1.0])
#     np.testing.assert_allclose(forces_2x["alignment"][0],
#                                2.0 * forces_1x["alignment"][0])
#
# def test_step_reads_the_alignment_radius():
#     """The same pair of birds, neighbours under one radius and not the other.
#
#     The wide case is PHYSICS itself rather than a hardcoded radius, so this
#     tests the configured value rather than an arbitrary number that happens to
#     exceed 25.
#     """
#     positions = np.array([[50.0, 50.0],
#                           [75.0, 50.0]])   # d = 25, inside the configured 30
#     velocities = np.array([[1.0, 0.0],
#                            [0.0, 1.0]])
#
#     _, _, wide = step(positions, velocities, PHYSICS)
#     _, _, narrow = step(positions, velocities, replace(PHYSICS, alignment_radius=10.0))
#
#     np.testing.assert_allclose(wide["alignment"][0], [-1.0, 1.0])
#     np.testing.assert_allclose(narrow["alignment"][0], [0.0, 0.0])
#
# def test_positions_wrap_the_lower_seam():
#     """Crossing x = 0 travelling in -x.
#
#     The suite already covers the upper seam (99.9 -> 0.4). The lower one
#     relies on Python's modulo returning a non-negative result for a positive
#     modulus - which is a language guarantee, not a universal one; C's fmod
#     would leave -0.4 here. `get_torus` assumes every position lives in
#     [0, bounds), so that assumption is worth pinning.
#     """
#     positions = np.array([[0.1, 50.0]])
#     velocities = np.array([[-10.0, 0.0]])   # capped to -5.0, so -0.5 per tick
#
#     new_positions, _, _ = step(positions, velocities, PHYSICS)
#
#     np.testing.assert_allclose(new_positions[0], [99.6, 50.0])
#     assert np.all(new_positions >= 0.0)
#
# def test_step_does_not_mutate_its_inputs():
#     """Double-buffering: every force for tick t comes from the state at t.
#
#     `step` currently builds fresh arrays, so this passes today. It is here
#     because the obvious optimisation later - doing `positions %= bounds` in
#     place instead of allocating - would compute some birds' forces against a
#     half-updated world. The forces would still be internally consistent, so
#     the reconstruction test would keep passing while the labels described a
#     world that never existed.
#     """
#     positions = np.array([[10.0, 10.0],
#                           [15.0, 10.0]])
#     velocities = np.array([[1.0, 0.0],
#                            [-1.0, 0.0]])
#     positions_before = positions.copy()
#     velocities_before = velocities.copy()
#
#     step(positions, velocities, PHYSICS)
#
#     np.testing.assert_array_equal(positions, positions_before)
#     np.testing.assert_array_equal(velocities, velocities_before)
#
#
# def test_step_reads_the_cohesion_weight():
#     positions = np.array([[50.0, 50.0],
#                           [60.0, 50.0]])
#     velocities = np.zeros((2, 2))
#
#     _, _, forces_1x = step(positions, velocities, replace(PHYSICS, w_coh=1.0))
#     _, _, forces_2x = step(positions, velocities, replace(PHYSICS, w_coh=2.0))
#
#     np.testing.assert_allclose(forces_1x["cohesion"][0], [0.25, 0.0])
#     np.testing.assert_allclose(forces_2x["cohesion"][0],
#                                2.0 * forces_1x["cohesion"][0])
#
#
# def test_step_reads_the_cohesion_radius():
#     """The same pair of birds, inside one radius and outside the other."""
#     positions = np.array([[50.0, 50.0],
#                           [80.0, 50.0]])   # 30 apart
#     velocities = np.zeros((2, 2))
#
#     _, _, wide = step(positions, velocities,
#                       replace(PHYSICS, cohesion_radius=40.0, w_coh=1.0))
#     _, _, narrow = step(positions, velocities,
#                         replace(PHYSICS, cohesion_radius=10.0, w_coh=1.0))
#
#     np.testing.assert_allclose(wide["cohesion"][0], [0.75, 0.0])
#     np.testing.assert_allclose(narrow["cohesion"][0], [0.0, 0.0])
#
#
# def test_step_records_all_three_rules_separately():
#     """The forces dict is the label set. Each key must carry its own rule.
#
#     A world arranged so that all three rules fire at once, with three
#     distinct answers: if any two keys ever hold the same array, the probes
#     would be trained to distinguish rules the simulator never separated.
#     """
#     positions = np.array([[50.0, 50.0],
#                           [55.0, 50.0],
#                           [50.0, 58.0]])
#     velocities = np.array([[1.0, 0.0],
#                            [0.0, 2.0],
#                            [-1.0, 1.0]])
#
#     _, _, forces = step(positions, velocities, PHYSICS)
#
#     for rule in ("separation", "alignment", "cohesion"):
#         assert np.any(forces[rule] != 0.0), f"{rule} did not fire"
#
#     assert not np.allclose(forces["separation"], forces["alignment"])
#     assert not np.allclose(forces["separation"], forces["cohesion"])
#     assert not np.allclose(forces["alignment"], forces["cohesion"])
#
#
# def test_step_reads_the_min_speed():
#     """A floor that step accepts and never forwards is invisible.
#
#     Two birds 70 units apart, outside every radius, so no rule fires and the
#     only thing that can change their velocity is the floor. With it off they
#     coast at speed 1; with it on they must be flying at min_speed.
#     """
#     positions = np.array([[10.0, 10.0], [60.0, 60.0]])
#     velocities = np.array([[1.0, 0.0], [-1.0, 0.0]])
#
#     _, drifting, _ = step(positions, velocities, PHYSICS)
#     _, cruising, _ = step(positions, velocities, PHYSICS_CRUISING)
#
#     np.testing.assert_allclose(drifting, velocities)
#     np.testing.assert_allclose(cruising, [[MIN_SPEED, 0.0], [-MIN_SPEED, 0.0]])
#
#
# def test_step_reads_the_dt():
#     """dt must reach both the velocity update and the position update.
#
#     Every other test in this suite runs at dt=0.1, so `step` ignoring
#     physics.dt and using the literal 0.1 was invisible until this existed. The
#     dataset config would then claim one timestep while the simulator ran
#     another, and every number in the file would look completely reasonable.
#     """
#     # neighbours in range, so the forces are non-zero and dt has to scale them
#     positions = np.array([[50.0, 50.0], [55.0, 50.0]])
#     velocities = np.zeros((2, 2))
#
#     p_slow, v_slow, f_slow = step(positions, velocities, replace(PHYSICS, dt=0.1))
#     p_fast, v_fast, f_fast = step(positions, velocities, replace(PHYSICS, dt=0.2))
#
#     # the forces themselves do not depend on dt - only how much of them is applied
#     np.testing.assert_allclose(f_slow["separation"], f_fast["separation"])
#     # twice the timestep, twice the velocity change from the same force
#     np.testing.assert_allclose(v_fast, 2.0 * v_slow)
#     # and the position update uses it too: dx = v_new * dt, so four times as far
#     np.testing.assert_allclose(p_fast - positions, 4.0 * (p_slow - positions))
#
#
# def test_step_reads_the_bounds():
#     """The world size must come from physics, not from a literal.
#
#     A single bird half a step from the seam of a 50-wide world. If `bounds`
#     were hardcoded to 100 it would sail past 50 and sit outside its own world,
#     where get_torus silently starts returning the wrong displacement for
#     everything.
#     """
#     small_world = replace(PHYSICS, bounds=50.0, separation_radius=20.0,
#                           alignment_radius=20.0, cohesion_radius=20.0)
#     positions = np.array([[49.9, 25.0]])
#     velocities = np.array([[5.0, 0.0]])
#
#     new_positions, _, _ = step(positions, velocities, small_world)
#
#     np.testing.assert_allclose(new_positions[0], [0.4, 25.0])
#
#
# def test_recorded_forces_match_the_rules_applied_to_the_input_state():
#     """The strongest statement this suite can make about the labels.
#
#     For every bird in a real flock, each recorded force must equal its own rule
#     function, at its own radius, times its own weight, evaluated on the state
#     that was passed *in*. That single assertion pins at once: the right rule in
#     the right key, the right constant reaching the right rule, bird i's force
#     stored at index i, and - the one nothing else covers - the labels
#     describing the world at tick t rather than the world at tick t+1.
#
#     That last one is the quietest possible label bug. Forces computed from the
#     updated positions would still satisfy the reconstruction invariant, because
#     that invariant only checks that the recorded numbers reproduce the velocity
#     update - not which world they were measured in.
#     """
#     physics = PHYSICS_CRUISING
#     positions, velocities = random_flock(seed=7, n_birds=12)
#
#     _, _, forces = step(positions, velocities, physics)
#
#     for i in range(len(positions)):
#         np.testing.assert_allclose(
#             forces["separation"][i],
#             physics.w_sep * calculate_separation(
#                 i, positions, physics.separation_radius, physics.bounds,
#                 physics.sep_force_cap),
#             err_msg=f"separation label wrong for bird {i}")
#         np.testing.assert_allclose(
#             forces["alignment"][i],
#             physics.w_ali * calculate_alignment(
#                 i, positions, velocities, physics.alignment_radius, physics.bounds),
#             err_msg=f"alignment label wrong for bird {i}")
#         np.testing.assert_allclose(
#             forces["cohesion"][i],
#             physics.w_coh * calculate_cohesion(
#                 i, positions, physics.cohesion_radius, physics.bounds),
#             err_msg=f"cohesion label wrong for bird {i}")
#
#     # premise guard: a frozen flock would satisfy all of the above trivially
#     for rule in ("separation", "alignment", "cohesion"):
#         assert np.linalg.norm(forces[rule], axis=1).max() > 0, f"{rule} never fired"
#
#
# # --- dtype: what survives being written to disk and read back ----------------
#
# def test_step_accepts_integer_input_without_truncating():
#     """Integer positions must not quantise the world to a lattice.
#
#     `np.zeros_like(positions)` without `dtype=float` inherits int64 from an
#     integer input, and every position then rounds to a whole number on every
#     tick. Birds would snap to a 1x1 grid, all three rules would be computed
#     from quantised distances, and nothing would raise - the flock would simply
#     be wrong in a way that looks like a rendering choice.
#     """
#     positions = np.array([[10, 10], [15, 10], [80, 80]])
#     velocities = np.array([[1, 0], [-1, 0], [0, 1]])
#     assert positions.dtype == np.int64 and velocities.dtype == np.int64
#
#     new_positions, new_velocities, forces = step(positions, velocities, PHYSICS)
#
#     assert new_positions.dtype == np.float64
#     assert new_velocities.dtype == np.float64
#     # bird 0 moves 1.0 * dt = 0.1 in x, plus a small push from its neighbour
#     assert new_positions[0][0] != int(new_positions[0][0]), (
#         "position landed on a whole number: the output array inherited an "
#         "integer dtype and the world has been quantised"
#     )
#
#
# def test_step_always_computes_in_float64():
#     """float32 in, float64 out - deliberately.
#
#     The dataset will be stored as float32 to keep it small, but the simulation
#     itself should not inherit that precision from whatever it was handed. This
#     pins the promotion so a future refactor cannot quietly make the physics
#     depend on the caller's dtype.
#     """
#     positions, velocities = random_flock(seed=0, n_birds=10)
#     positions, velocities = positions.astype(np.float32), velocities.astype(np.float32)
#
#     new_positions, new_velocities, forces = step(positions, velocities, PHYSICS_CRUISING)
#
#     assert new_positions.dtype == np.float64
#     assert new_velocities.dtype == np.float64
#     for rule, force in forces.items():
#         assert force.dtype == np.float64, f"{rule} came back as {force.dtype}"
#
#
# def test_the_reconstruction_invariant_is_exact_in_float64():
#     """Not approximately equal - bit for bit identical.
#
#     Every other reconstruction test allows a tolerance. Measured over 200 ticks
#     of a real flock the error is exactly zero, because `step` and the
#     reconstruction perform the same operations in the same order on the same
#     float64 values. Asserting equality rather than closeness means any future
#     reordering that introduces rounding shows up here first, while it is still
#     one line of blame rather than a drift in a trained result.
#     """
#     physics = PHYSICS_CRUISING
#     positions, velocities = random_flock(seed=3, n_birds=15)
#
#     for tick in range(50):
#         old_velocities = velocities.copy()
#         positions, velocities, forces = step(positions, velocities, physics)
#         total = forces["separation"] + forces["alignment"] + forces["cohesion"]
#
#         for i in range(len(positions)):
#             np.testing.assert_array_equal(
#                 velocity_capper(old_velocities[i] + total[i] * physics.dt,
#                                 physics.max_speed, physics.min_speed),
#                 velocities[i],
#                 err_msg=f"reconstruction is no longer exact for bird {i} at tick {tick}")
#
#
# def test_the_reconstruction_invariant_survives_float32_storage():
#     """What the invariant is actually worth once the labels are on disk.
#
#     The dataset stores velocities and forces as float32. Reading them back and
#     re-deriving the velocity update can no longer be exact - measured worst
#     case 2.9e-07, about 2.4x float32 epsilon. Anything materially larger than
#     that means precision is being lost somewhere other than the cast, and a
#     test asserting exact equality against reloaded data would fail for a reason
#     that has nothing to do with the simulator.
#     """
#     physics = PHYSICS_CRUISING
#     positions, velocities = random_flock(seed=3, n_birds=15)
#     worst = 0.0
#
#     for _ in range(100):
#         old_velocities = velocities.copy()
#         positions, velocities, forces = step(positions, velocities, physics)
#         total = forces["separation"] + forces["alignment"] + forces["cohesion"]
#
#         def stored(a):                      # write to disk as float32, read back
#             return a.astype(np.float32).astype(np.float64)
#
#         rebuilt = np.array([
#             velocity_capper(stored(old_velocities)[i] + stored(total)[i] * physics.dt,
#                             physics.max_speed, physics.min_speed)
#             for i in range(len(positions))])
#         worst = max(worst, np.abs(rebuilt - stored(velocities)).max())
#
#     assert worst < 1e-6, f"float32 round trip lost {worst:.3e}, more than the cast alone"
#     assert worst > 0.0, (
#         "premise failed: the float32 round trip was lossless, so this test did "
#         "not exercise the precision loss it exists to bound"
#     )

# ==========================================================================
# simulator/test_rollout.py
# ==========================================================================
# """The simulator run forwards, rather than one tick at a time.
#
# Every other file in this suite checks a single call. These check what happens
# over hundreds of them, which is the regime the dataset is actually generated
# in and where a different class of bug lives: a NaN that appears on tick 300
# and spreads, a position that drifts outside the world, an integrator that
# quietly pumps energy into the flock until every bird is pinned at the speed
# cap. None of those can be caught by looking at one tick.
# """
#
# from dataclasses import replace
#
# import numpy as np
#
# from boids import Physics, step, velocity_capper
# from world import BOUNDS, DT, MAX_SPEED, PHYSICS, PHYSICS_CRUISING, random_flock
#
# TICKS = 200
# N_BIRDS = 20
#
#
# def test_no_nan_ever_appears():
#     """The failure this whole project is most exposed to.
#
#     A NaN in a force is not an error - it propagates into the velocity, then
#     the position, then every subsequent tick, and the run completes. The
#     dataset would be written, the model would train on it, and the labels
#     would be garbage from some arbitrary tick onwards.
#     """
#     positions, velocities = random_flock(seed=0, n_birds=N_BIRDS)
#
#     for tick in range(TICKS):
#         positions, velocities, forces = step(positions, velocities, PHYSICS)
#         assert not np.isnan(positions).any(), f"NaN position at tick {tick}"
#         assert not np.isnan(velocities).any(), f"NaN velocity at tick {tick}"
#         for rule, force in forces.items():
#             assert not np.isnan(force).any(), f"NaN {rule} force at tick {tick}"
#
#
# def test_positions_stay_inside_the_world_for_every_tick():
#     """`pos %= bounds` every tick is the invariant get_torus depends on.
#
#     If a position ever escapes [0, bounds), the minimum-image convention
#     silently starts returning the wrong displacement and every rule that uses
#     it is computed against a world that has come apart.
#     """
#     positions, velocities = random_flock(seed=1, n_birds=N_BIRDS)
#
#     for tick in range(TICKS):
#         positions, velocities, _ = step(positions, velocities, PHYSICS)
#         assert np.all(positions >= 0.0), f"bird left the world at tick {tick}"
#         assert np.all(positions < BOUNDS), f"bird left the world at tick {tick}"
#
#
# def test_speed_never_exceeds_the_cap():
#     positions, velocities = random_flock(seed=2, n_birds=N_BIRDS)
#
#     for tick in range(TICKS):
#         positions, velocities, _ = step(positions, velocities, PHYSICS)
#         fastest = np.linalg.norm(velocities, axis=1).max()
#         assert fastest <= MAX_SPEED + 1e-9, (
#             f"speed {fastest} exceeded the cap at tick {tick}"
#         )
#
#
# def _assert_reconstruction_holds(physics, seed, ticks=TICKS):
#     """Run a flock forwards, re-deriving every velocity from recorded forces.
#
#     Returns the number of ticks on which the speed cap actually engaged, so
#     the caller can guard its own premise.
#     """
#     positions, velocities = random_flock(seed=seed, n_birds=N_BIRDS)
#     ticks_where_the_cap_fired = 0
#
#     for tick in range(ticks):
#         old_velocities = velocities.copy()
#         positions, velocities, forces = step(positions, velocities, physics)
#
#         totals = forces["separation"] + forces["alignment"] + forces["cohesion"]
#         uncapped = old_velocities + totals * physics.dt
#         if np.linalg.norm(uncapped, axis=1).max() > physics.max_speed:
#             ticks_where_the_cap_fired += 1
#
#         for i in range(N_BIRDS):
#             np.testing.assert_allclose(
#                 velocity_capper(uncapped[i], physics.max_speed, physics.min_speed),
#                 velocities[i],
#                 err_msg=f"reconstruction broke for bird {i} at tick {tick}",
#             )
#
#     return ticks_where_the_cap_fired
#
#
# def test_the_reconstruction_invariant_holds_on_every_tick():
#     """The load-bearing test of this project, run forwards.
#
#     The single-tick version proves the recorded forces reproduce one velocity
#     update in a hand-built world. This proves it for every bird on every one
#     of two hundred ticks of a real flock - through seam crossings, through
#     birds that lose all their neighbours, through the whole settling process.
#
#     If this ever fails, the numbers in the forces dict are not labels for this
#     simulation, and every probe and ablation result downstream is fitted to
#     something that never happened.
#     """
#     _assert_reconstruction_holds(PHYSICS, seed=3)
#
#
# def test_the_reconstruction_invariant_holds_when_the_cap_fires_constantly():
#     """The same invariant in the regime where forces do not simply sum.
#
#     Once the speed cap engages, the relationship is cap(v + sum(f)*dt) == v',
#     not v + sum(f)*dt == v'. A settled flock almost never reaches its cap -
#     measured, 1 tick in 200 - so an ordinary rollout barely tests this at all.
#     Strong separation against a low cap saturates it on every single tick.
#     """
#     physics = replace(PHYSICS, max_speed=0.3, w_sep=5.0)
#
#     ticks_where_the_cap_fired = _assert_reconstruction_holds(physics, seed=3)
#
#     assert ticks_where_the_cap_fired == TICKS, (
#         f"premise failed: the cap fired on only {ticks_where_the_cap_fired} of "
#         f"{TICKS} ticks, so this barely tested the capped regime"
#     )
#
#
# def test_the_flock_settles_rather_than_being_pinned_at_the_cap():
#     """A flock that has found its shape should be moving gently, not flat out.
#
#     If the integrator pumped energy in - which plain forward Euler does to a
#     spring force like cohesion - every bird would end up saturated at the
#     speed cap and stay there. Measured: mean speed falls from ~3.7 to ~0.6 and
#     the fraction of birds at the cap goes to zero.
#     """
#     positions, velocities = random_flock(seed=4, n_birds=N_BIRDS)
#     for _ in range(TICKS):
#         positions, velocities, _ = step(positions, velocities, PHYSICS)
#
#     speeds = np.linalg.norm(velocities, axis=1)
#     at_the_cap = (speeds > MAX_SPEED - 1e-9).mean()
#
#     assert at_the_cap < 0.1, (
#         f"{at_the_cap:.0%} of birds are pinned at the speed cap; the flock is "
#         "being driven rather than settling"
#     )
#
#
# def test_the_cohesion_spring_does_not_gain_energy():
#     """Cohesion is Hooke's law, and springs are where integrators go wrong.
#
#     Two birds, cohesion only, oscillating about their shared centre of mass.
#     Under plain forward Euler the amplitude would grow every period until the
#     speed cap took over. It does not, because step() advances position using
#     the *new* velocity - semi-implicit (symplectic) Euler - which conserves
#     energy for oscillators. That one line is load-bearing and this pins it.
#     """
#     physics = replace(PHYSICS, w_sep=0.0, w_ali=0.0, w_coh=1.0)
#     positions = np.array([[45.0, 50.0], [55.0, 50.0]])
#     velocities = np.zeros((2, 2))
#
#     separations = []
#     for _ in range(4000):
#         positions, velocities, _ = step(positions, velocities, physics)
#         offset = (positions[1] - positions[0] + BOUNDS / 2) % BOUNDS - BOUNDS / 2
#         separations.append(np.linalg.norm(offset))
#     separations = np.array(separations)
#
#     early = separations[:400].max()
#     late = separations[-400:].max()
#
#     # the cap must stay out of the way or it would mask any growth
#     assert np.linalg.norm(velocities, axis=1).max() < MAX_SPEED
#     assert late < early * 1.01, (
#         f"oscillation grew from {early:.3f} to {late:.3f} over 4000 ticks - "
#         "check that positions advance using the new velocity, not the old"
#     )
#
#
# def test_a_rollout_is_reproducible():
#     """Same seed, same physics, identical trajectory - bit for bit.
#
#     Without this the dataset cannot be regenerated from its own metadata, and
#     a checkpoint sweep cannot be compared against a rerun of the world it was
#     trained on.
#     """
#     first_positions, first_velocities = random_flock(seed=5, n_birds=10)
#     second_positions, second_velocities = random_flock(seed=5, n_birds=10)
#
#     for _ in range(50):
#         first_positions, first_velocities, first_forces = step(
#             first_positions, first_velocities, PHYSICS)
#         second_positions, second_velocities, second_forces = step(
#             second_positions, second_velocities, PHYSICS)
#
#     np.testing.assert_array_equal(first_positions, second_positions)
#     np.testing.assert_array_equal(first_velocities, second_velocities)
#     for rule in first_forces:
#         np.testing.assert_array_equal(first_forces[rule], second_forces[rule])
#
#
# def test_two_birds_beyond_every_radius_just_drift():
#     """No neighbours, no forces, pure translation.
#
#     The two birds here are 70.7 units apart on the torus, outside all three
#     radii. Their velocities must not change at all, and their positions must
#     advance by exactly v*dt each tick. Any drift means a rule is firing when
#     nothing is in range.
#     """
#     positions = np.array([[10.0, 10.0], [60.0, 60.0]])
#     velocities = np.array([[1.0, 0.5], [-2.0, 0.0]])
#     starting_velocities = velocities.copy()
#     expected = positions.copy()
#
#     for _ in range(100):
#         positions, velocities, forces = step(positions, velocities, PHYSICS)
#         expected = (expected + starting_velocities * DT) % BOUNDS
#
#     np.testing.assert_allclose(velocities, starting_velocities)
#     np.testing.assert_allclose(positions, expected)
#     for rule, force in forces.items():
#         np.testing.assert_array_equal(force, np.zeros((2, 2)), err_msg=rule)
#
#
# def test_the_reconstruction_invariant_holds_with_the_speed_floor():
#     """The load-bearing invariant under the new clamp.
#
#     Adding the floor changed the function on the outside of
#     `clamp(v + sum(f)*dt) == v'`. The recorded forces must still reproduce the
#     update, on every bird on every tick, with the floor engaged.
#     """
#     _assert_reconstruction_holds(PHYSICS_CRUISING, seed=3)
#
#
# def test_a_cruising_flock_keeps_moving():
#     """Why the floor exists at all.
#
#     Without it, alignment steers each bird toward its neighbours' mean
#     *vector*, and the mean of scattered vectors is short - so the flock
#     organises its headings and brakes to a crawl in the same motion. Measured
#     without the floor: mean speed falls to about 0.27 and stays there.
#     """
#     positions, velocities = random_flock(seed=0, n_birds=N_BIRDS)
#     for _ in range(TICKS):
#         positions, velocities, _ = step(positions, velocities, PHYSICS_CRUISING)
#     cruising = np.linalg.norm(velocities, axis=1)
#
#     positions, velocities = random_flock(seed=0, n_birds=N_BIRDS)
#     for _ in range(TICKS):
#         positions, velocities, _ = step(positions, velocities, PHYSICS)
#     drifting = np.linalg.norm(velocities, axis=1)
#
#     assert cruising.min() >= PHYSICS_CRUISING.min_speed - 1e-9
#     assert cruising.mean() < PHYSICS_CRUISING.min_speed * 1.5
#     assert drifting.mean() < PHYSICS_CRUISING.min_speed / 2, (
#         "premise failed: the flock did not slow down without the floor, so "
#         "this comparison shows nothing"
#     )
#
#
# def test_speed_stays_between_the_floor_and_the_cap_on_every_tick():
#     """The screenshot, as an assertion.
#
#     Random headings, 500 ticks, floor engaged: every bird must sit inside
#     [min_speed, max_speed] at every single tick, not merely at the end. The
#     picture shows this as every arrow being the same length from tick 25
#     onwards; this is that claim in a form that fails loudly.
#
#     It would also catch a bird stalling to exactly zero, which the floor cannot
#     rescue because a zero vector has no direction - measured to happen only in
#     flocks too small to be flocks, and this pins that it stays that way.
#     """
#     physics = PHYSICS_CRUISING
#     positions, velocities = random_flock(seed=11, n_birds=N_BIRDS)
#     floor_engaged = 0
#
#     for tick in range(500):
#         positions, velocities, _ = step(positions, velocities, physics)
#         speeds = np.linalg.norm(velocities, axis=1)
#         assert speeds.min() >= physics.min_speed - 1e-9, (
#             f"bird below the floor at tick {tick}: {speeds.min()}")
#         assert speeds.max() <= physics.max_speed + 1e-9, (
#             f"bird above the cap at tick {tick}: {speeds.max()}")
#         floor_engaged += int(np.isclose(speeds.min(), physics.min_speed))
#
#     # Measured at this density: the floor holds at least one bird on about
#     # 30% of ticks. At 60 birds it is every tick; at 20 the flock is sparse
#     # enough that many birds drift above it on their own.
#     assert floor_engaged > 50, (
#         f"premise failed: the floor only engaged on {floor_engaged} of 500 "
#         "ticks, so this mostly tested a flock that never needed it"
#     )
#
#
# def test_the_shipped_defaults_produce_a_sane_rollout():
#     """Physics() with no arguments is what a careless caller gets.
#
#     Since world.py began deriving from Physics() this is nearly what
#     PHYSICS_CRUISING already is, so the gap this test was written to cover has
#     mostly closed. It stays because it constructs Physics() literally: if a
#     future default were changed without world.py being re-read, this is the
#     test that runs against the new value rather than a remembered one.
#     """
#     physics = Physics()
#     positions, velocities = random_flock(seed=13, n_birds=N_BIRDS)
#
#     for tick in range(TICKS):
#         positions, velocities, forces = step(positions, velocities, physics)
#         speeds = np.linalg.norm(velocities, axis=1)
#         assert not np.isnan(positions).any(), f"NaN position at tick {tick}"
#         assert np.all(positions >= 0.0) and np.all(positions < physics.bounds)
#         assert physics.min_speed - 1e-9 <= speeds.min()
#         assert speeds.max() <= physics.max_speed + 1e-9
#         for rule, force in forces.items():
#             assert not np.isnan(force).any(), f"NaN {rule} force at tick {tick}"

# ==========================================================================
# dataset/test_generate.py
# ==========================================================================
# """The contract for `generate_clip()` — written before it exists.
#
# This file is the specification. `generate.py` does not exist yet, so the whole
# module skips; the moment it does, these become real and most of them will be
# red. Build into them.
#
#     generate_clip(physics, *, n_slots, n_live, n_ticks, seed) -> dict
#
# Returns one clip's arrays in memory. No file I/O, no config, no summary — those
# are the writer's job and get their own tests.
#
#     positions    (n_ticks + 1, n_slots, 2)  float64   state at ticks 0..n_ticks
#     velocities   (n_ticks + 1, n_slots, 2)  float64
#     separation   (n_ticks,     n_slots, 2)  float64   force on state[t] -> [t+1]
#     alignment    (n_ticks,     n_slots, 2)  float64
#     cohesion     (n_ticks,     n_slots, 2)  float64
#     mask         (n_ticks + 1, n_slots)     bool      slot holds a live entity
#     kind         (n_slots,)                 int8      0 = bird
#     settle_tick  int                        smallest index into `velocities`
#                                             with polarisation > 0.99, else -1
#
# Four decisions are baked in here. Say so and I will change them:
#
# 1. **A flat dict**, keyed exactly as the plan's storage layout names the arrays,
#    so the same word means the same thing in memory, on disk and in the summary.
# 2. **float64 in memory.** The plan stores float32; that cast belongs to the
#    writer, in one place. Keeping the clip lossless means the tick-alignment test
#    below can assert *exact* equality rather than a tolerance, which is a much
#    stronger claim.
# 3. **Keyword-only sizing arguments.** `n_slots`, `n_live` and `n_ticks` are three
#    ints in a row. Passed positionally, swapping any two runs perfectly and yields
#    a wrong dataset — the same failure that put eleven positional floats into
#    `step` and cost a refactor to undo.
# 4. **`kind` has no tick axis.** Nothing in `SPEC.md` changes an entity's type
#    mid-clip; `mask` carries the tick axis because §6.2 drops a bird mid-flight.
# """
#
# import numpy as np
# import pytest
#
# from boids import (calculate_alignment, calculate_cohesion,
#                    calculate_separation, velocity_capper)
# from world import PHYSICS_CRUISING
#
# generate = pytest.importorskip(
#     "generate",
#     reason="generate.py does not exist yet — this suite is its specification")
#
# PHYSICS = PHYSICS_CRUISING      # the shipped config, min_speed = 2.0
# N_SLOTS = 12
# N_TICKS = 20                    # short on purpose: contract tests, not rollouts
#
#
# def clip(n_live=N_SLOTS, n_ticks=N_TICKS, seed=0, n_slots=N_SLOTS, physics=PHYSICS):
#     return generate.generate_clip(physics, n_slots=n_slots, n_live=n_live,
#                                   n_ticks=n_ticks, seed=seed)
#
#
# def polarisation(velocities):
#     """Length of the mean unit heading: 0 is chaos, 1 is one flock."""
#     speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
#     return np.linalg.norm((velocities / np.where(speeds == 0, 1, speeds)).mean(axis=0))
#
#
# # --- polarisation ------------------------------------------------------------
#
# def test_polarisation_is_one_when_every_bird_flies_the_same_way():
#     v = np.array([[3.0, 0.0], [3.0, 0.0], [3.0, 0.0]])
#
#     assert generate.polarisation(v) == pytest.approx(1.0)
#
#
# def test_polarisation_ignores_speed_entirely():
#     """The definition is the mean *unit* heading, so speed must not vote.
#
#     Same three headings as above at wildly different speeds. If this returns
#     anything but 1.0, the function is averaging velocities rather than
#     directions.
#     """
#     v = np.array([[2.0, 0.0], [5.0, 0.0], [2.5, 0.0]])
#
#     assert generate.polarisation(v) == pytest.approx(1.0)
#
#
# def test_polarisation_is_zero_for_two_birds_flying_at_each_other():
#     """The regression test for a bug this function actually had.
#
#     The first implementation was |mean velocity| / mean speed, which lets a fast
#     bird outvote a slow one: these two score 0.43 under it and 0.0 under the
#     definition. Two birds flying directly at each other are not 43% ordered.
#     """
#     v = np.array([[5.0, 0.0], [-2.0, 0.0]])
#
#     assert generate.polarisation(v) == pytest.approx(0.0)
#
#
# def test_polarisation_of_perpendicular_headings():
#     """A value that is neither 0 nor 1, so the scale is checked and not just its ends."""
#     v = np.array([[4.0, 0.0], [0.0, 1.0]])
#
#     assert generate.polarisation(v) == pytest.approx(1 / np.sqrt(2))
#
#
# def test_polarisation_survives_a_stationary_bird():
#     """Normalising divides by each bird's speed, and a still bird's speed is 0.
#
#     min_speed keeps this out of generated clips, but the function is called on
#     hand-built flocks too, and a NaN here would propagate silently through every
#     mean downstream rather than raising.
#     """
#     v = np.array([[3.0, 0.0], [0.0, 0.0]])
#     result = generate.polarisation(v)
#
#     assert not np.isnan(result)
#     assert result == pytest.approx(0.5), "the still bird should count as no vote, not as a heading"
#
#
# def test_polarisation_does_not_depend_on_flock_orientation():
#     """Rotating the whole flock cannot change how ordered it is."""
#     v = np.array([[2.0, 1.0], [3.0, -1.0], [2.5, 0.5]])
#     angle = 0.7
#     rotation = np.array([[np.cos(angle), -np.sin(angle)],
#                          [np.sin(angle), np.cos(angle)]])
#
#     assert generate.polarisation(v @ rotation.T) == pytest.approx(generate.polarisation(v))
#
#
# # --- the shape contract ------------------------------------------------------
#
# def test_the_shape_contract():
#     """One more state than there are forces, and the asymmetry stays visible.
#
#     n_ticks steps produce n_ticks force sets and n_ticks + 1 states. Padding the
#     force arrays to match would fabricate a label for tick 0, where no force has
#     been computed yet.
#     """
#     c = clip()
#
#     assert c["positions"].shape == (N_TICKS + 1, N_SLOTS, 2)
#     assert c["velocities"].shape == (N_TICKS + 1, N_SLOTS, 2)
#     for rule in ("separation", "alignment", "cohesion"):
#         assert c[rule].shape == (N_TICKS, N_SLOTS, 2), rule
#     assert c["mask"].shape == (N_TICKS + 1, N_SLOTS)
#     assert c["kind"].shape == (N_SLOTS,)
#     assert isinstance(int(c["settle_tick"]), int)
#
#
# def test_arrays_are_float64_in_memory():
#     """float32 is the writer's decision, made once, at the point of saving.
#
#     Casting here would lose precision before anything has been written, and
#     would weaken the tick-alignment test from exact equality to a tolerance.
#     """
#     c = clip()
#
#     for key in ("positions", "velocities", "separation", "alignment", "cohesion"):
#         assert c[key].dtype == np.float64, f"{key} is {c[key].dtype}"
#     assert c["mask"].dtype == np.bool_
#     assert c["kind"].dtype == np.int8
#
#
# def test_more_live_birds_than_slots_is_refused():
#     """Nowhere to put them. Better a loud failure than a truncated flock."""
#     with pytest.raises((ValueError, AssertionError)):
#         clip(n_live=N_SLOTS + 1, n_slots=N_SLOTS)
#
#
# def test_the_sizing_arguments_are_keyword_only():
#     """Three ints in a row is a transposition waiting to happen.
#
#     `generate_clip(physics, 12, 10, 300, 0)` and `(physics, 10, 12, 300, 0)`
#     both run. One of them silently produces a clip with more live birds than
#     slots, or a differently shaped world. Keywords make it impossible rather
#     than unlikely.
#     """
#     with pytest.raises(TypeError):
#         generate.generate_clip(PHYSICS, N_SLOTS, N_SLOTS, N_TICKS, 0)
#
#
# # --- the two invariants that can silently corrupt every label ----------------
#
# def test_forces_produce_the_next_velocity_exactly():
#     """Tick alignment, stated as an equation.
#
#     `forces[t]` must be what acted on `velocities[t]` to give `velocities[t+1]`.
#     Record them one iteration apart and every label in the dataset describes the
#     wrong state — and the result still looks entirely plausible, because the
#     physics is self-consistent, just shifted.
#
#     Exact rather than approximate: same operations, same order, same float64
#     values, so any drift here means something was recomputed rather than
#     recorded.
#     """
#     c = clip()
#     total = c["separation"] + c["alignment"] + c["cohesion"]
#
#     for t in range(N_TICKS):
#         for i in range(N_SLOTS):
#             np.testing.assert_array_equal(
#                 velocity_capper(c["velocities"][t][i] + total[t][i] * PHYSICS.dt,
#                                 PHYSICS.max_speed, PHYSICS.min_speed),
#                 c["velocities"][t + 1][i],
#                 err_msg=f"tick {t}, slot {i}: forces do not produce the next velocity")
#
#
# def test_forces_are_the_rules_applied_to_the_state_at_that_tick():
#     """The strongest statement available about the labels.
#
#     Each recorded force must equal its own rule function, at its own radius,
#     times its own weight, evaluated on the state stored at the *same* index.
#     That pins the rule, the constant, the weight, the slot index and the tick
#     together. Nothing else can: forces taken from the following state still
#     satisfy the equation above, because that only checks self-consistency.
#     """
#     c = clip()
#     ph = PHYSICS
#
#     for t in (0, N_TICKS // 2, N_TICKS - 1):
#         positions, velocities = c["positions"][t], c["velocities"][t]
#         for i in range(N_SLOTS):
#             np.testing.assert_allclose(
#                 c["separation"][t][i],
#                 ph.w_sep * calculate_separation(i, positions, ph.separation_radius,
#                                                 ph.bounds, ph.sep_force_cap),
#                 err_msg=f"separation label wrong at tick {t}, slot {i}")
#             np.testing.assert_allclose(
#                 c["alignment"][t][i],
#                 ph.w_ali * calculate_alignment(i, positions, velocities,
#                                                ph.alignment_radius, ph.bounds),
#                 err_msg=f"alignment label wrong at tick {t}, slot {i}")
#             np.testing.assert_allclose(
#                 c["cohesion"][t][i],
#                 ph.w_coh * calculate_cohesion(i, positions, ph.cohesion_radius,
#                                               ph.bounds),
#                 err_msg=f"cohesion label wrong at tick {t}, slot {i}")
#
#     for rule in ("separation", "alignment", "cohesion"):
#         assert np.linalg.norm(c[rule], axis=2).max() > 0, f"{rule} never fired"
#
#
# # --- the initial state must be one the simulator could have produced --------
#
# def test_initial_velocities_are_legal():
#     """Tick 0 is a state, and it has to obey the same clamp as every other.
#
#     `tests/world.py:random_flock` draws velocity components uniform in
#     ±max_speed, which puts 33.2% of its birds outside `[min_speed, max_speed]`
#     (200 seeds × 12 birds). That is deliberate there — it exercises the clamp.
#     Here it would write a first row the simulator can never reach.
#
#     Draw a heading uniform in [0, 2π) and a speed uniform in the clamp instead.
#     """
#     speeds = np.linalg.norm(clip()["velocities"][0], axis=1)
#
#     assert speeds.min() >= PHYSICS.min_speed - 1e-12
#     assert speeds.max() <= PHYSICS.max_speed + 1e-12
#
#
# def test_headings_are_not_all_the_same_at_tick_zero():
#     """A premise guard for the test above.
#
#     Clamping every initial speed to exactly `min_speed` would satisfy the
#     previous test while producing a degenerate flock. Speeds should vary, and
#     headings should cover the circle rather than clustering.
#     """
#     velocities = clip(n_ticks=1)["velocities"][0]
#
#     assert np.std(np.linalg.norm(velocities, axis=1)) > 0.1
#     assert polarisation(velocities) < 0.9
#
#
# def test_the_world_invariants_hold_at_every_tick():
#     c = clip()
#     live = c["mask"]
#
#     for t in range(N_TICKS + 1):
#         positions = c["positions"][t][live[t]]
#         speeds = np.linalg.norm(c["velocities"][t][live[t]], axis=1)
#         assert np.all(positions >= 0.0) and np.all(positions < PHYSICS.bounds), t
#         assert speeds.min() >= PHYSICS.min_speed - 1e-9, t
#         assert speeds.max() <= PHYSICS.max_speed + 1e-9, t
#
#
# def test_no_nan_in_live_slots():
#     c = clip()
#     live = c["mask"]
#
#     assert not np.isnan(c["positions"][live]).any()
#     assert not np.isnan(c["velocities"][live]).any()
#     for rule in ("separation", "alignment", "cohesion"):
#         assert not np.isnan(c[rule][live[:-1]]).any(), rule
#
#
# # --- slots, masks and the birds that are not there --------------------------
#
# def test_every_slot_is_live_when_n_live_equals_n_slots():
#     c = clip(n_live=N_SLOTS)
#
#     assert c["mask"].all()
#     np.testing.assert_array_equal(c["kind"], np.zeros(N_SLOTS, dtype=np.int8))
#
#
# def test_dead_slots_are_nan_and_exactly_complement_the_mask():
#     """NaN in dead slots is a checksum on the mask, not decoration.
#
#     With zeros you cannot tell a dead slot from a bird at the origin, and 0.0
#     and -1.0 are both legal velocities. With NaN the two encodings must agree
#     exactly, and disagreeing is a test failure rather than a silent one.
#     """
#     c = clip(n_live=8)
#     dead = ~c["mask"]
#
#     assert dead.any(), "premise failed: nothing was masked off"
#     np.testing.assert_array_equal(np.isnan(c["positions"][..., 0]), dead)
#     np.testing.assert_array_equal(np.isnan(c["velocities"][..., 0]), dead)
#     for rule in ("separation", "alignment", "cohesion"):
#         np.testing.assert_array_equal(np.isnan(c[rule][..., 0]), dead[:-1], rule)
#
#
# def test_masked_slots_exert_no_force_on_live_birds():
#     """The quietest way to produce a plausible dataset of the wrong physics.
#
#     `step` uses `N = len(positions)`, so a dead slot handed to it is simply a
#     bird — it pushes on its neighbours and enters their averages, and nothing
#     raises. Simulate an (n_live, 2) array and scatter into slots afterwards.
#
#     This also pins the initialiser's draw *size*: the two clips below only match
#     if the random stream is identical, which requires drawing exactly `n_live`
#     positions and velocities rather than `n_slots` with some discarded.
#     """
#     wide = clip(n_live=9, n_slots=12, seed=7)
#     narrow = clip(n_live=9, n_slots=9, seed=7)
#
#     live = wide["mask"][0]
#     np.testing.assert_array_equal(wide["positions"][:, live, :], narrow["positions"])
#     np.testing.assert_array_equal(wide["velocities"][:, live, :], narrow["velocities"])
#     for rule in ("separation", "alignment", "cohesion"):
#         np.testing.assert_array_equal(wide[rule][:, live, :], narrow[rule], rule)
#
#
# # --- reproducibility ---------------------------------------------------------
#
# def test_the_same_seed_reproduces_the_clip_bit_for_bit():
#     """Without this the dataset cannot be regenerated from its own metadata."""
#     first, second = clip(seed=99), clip(seed=99)
#
#     for key in ("positions", "velocities", "separation", "alignment", "cohesion"):
#         np.testing.assert_array_equal(first[key], second[key], err_msg=key)
#     np.testing.assert_array_equal(first["mask"], second["mask"])
#
#
# def test_a_seed_sequence_is_accepted_and_distinguishes_clips():
#     """Clips are seeded by `SeedSequence(entropy=base, spawn_key=(i,))`.
#
#     Children of `spawn()` all inherit the parent's `.entropy`, so seeding from
#     that alone gives every clip the same stream. The spawn_key is what
#     distinguishes them, and it must survive being passed in here.
#     """
#     base = 4242
#     a = clip(seed=np.random.SeedSequence(entropy=base, spawn_key=(0,)))
#     b = clip(seed=np.random.SeedSequence(entropy=base, spawn_key=(1,)))
#     a_again = clip(seed=np.random.SeedSequence(entropy=base, spawn_key=(0,)))
#
#     np.testing.assert_array_equal(a["positions"], a_again["positions"])
#     assert not np.array_equal(a["positions"], b["positions"])
#
#
# def test_different_seeds_give_different_clips():
#     assert not np.array_equal(clip(seed=1)["positions"], clip(seed=2)["positions"])
#
#
# # --- the settling tick -------------------------------------------------------
#
# def test_settle_tick_matches_a_recomputation_from_the_stored_velocities():
#     """Defined by polarisation, not by a force threshold.
#
#     Polarisation is dimensionless and bounded, so it survives any future
#     reweighting of the rules; an absolute force threshold would need retuning.
#     """
#     c = clip(n_ticks=400)
#     live = c["mask"][0]
#
#     expected = -1
#     for t in range(len(c["velocities"])):
#         if polarisation(c["velocities"][t][live]) > 0.99:
#             expected = t
#             break
#
#     assert int(c["settle_tick"]) == expected
#
#
# def test_settle_tick_is_minus_one_when_the_flock_never_orders():
#     """A clip too short to settle must say so, not report its last tick."""
#     c = clip(n_ticks=2)
#     live = c["mask"][0]
#
#     assert max(polarisation(c["velocities"][t][live])
#                for t in range(len(c["velocities"]))) <= 0.99, (
#         "premise failed: this clip did settle, so it cannot test the -1 case")
#     assert int(c["settle_tick"]) == -1

# ==========================================================================
# dataset/test_dataset.py
# ==========================================================================
# """The contract for the dataset writer — written before it exists.
#
# `generate_clip()` is done and green. This file specifies everything that turns
# clips into a dataset on disk, and it skips as one unit until `build_dataset`
# exists. Build into it.
#
#     build_dataset(physics, *, n_clips, n_slots, n_live, n_ticks,
#                   base_seed, out_dir, allow_dirty=False) -> pathlib.Path
#
#     clip_seed(base_seed, index) -> np.random.SeedSequence
#     git_is_dirty() -> bool
#     git_provenance() -> dict   {"commit": str, "tag": str, "dirty": bool}
#
# `build_dataset` writes exactly two files into `out_dir` and returns it:
#
#     clips.npz     positions    (n_clips, n_ticks+1, n_slots, 2)  float32
#                   velocities   (n_clips, n_ticks+1, n_slots, 2)  float32
#                   separation   (n_clips, n_ticks,   n_slots, 2)  float32
#                   alignment    (n_clips, n_ticks,   n_slots, 2)  float32
#                   cohesion     (n_clips, n_ticks,   n_slots, 2)  float32
#                   mask         (n_clips, n_ticks+1, n_slots)     bool
#                   kind         (n_clips, n_slots)                int8
#                   settle_tick  (n_clips,)                        int32
#
#     config.json   {"physics": {...}, "generation": {...}, "provenance": {...}}
#
# Four decisions are baked in here. Say so and I will change them:
#
# 1. **No `seeds` column.** A clip is identified by `(base_seed, its index)`:
#    the base seed is in `config.json` and the index is the array index it already
#    sits at. The plan left the column optional on the condition that it hold
#    something that actually regenerates a clip — and a column that must *not*
#    contain `SeedSequence.entropy`, while looking exactly like it should, is a
#    trap for whoever reads this next. `clip_seed` is the regeneration path.
# 2. **The float32 cast lives here and only here.** `generate_clip` stays float64,
#    so the exact tick-alignment test keeps its exactness and this file measures
#    what the cast costs.
# 3. **Fixed filenames**, so nothing downstream takes a path template as an
#    argument.
# 4. **`build_dataset` returns the directory, not the arrays.** Reading the
#    dataset back is `np.load`; a function that both writes and returns would let
#    every check downstream quietly test the in-memory copy instead of the file.
# """
#
# import json
# import subprocess
#
# import numpy as np
# import pytest
#
# from boids import Physics, velocity_capper
# from world import PHYSICS_CRUISING
#
# generate = pytest.importorskip("generate")
#
# pytestmark = pytest.mark.skipif(
#     not hasattr(generate, "build_dataset"),
#     reason="generate.build_dataset() does not exist yet — this suite is its "
#            "specification")
#
# PHYSICS = PHYSICS_CRUISING
# N_CLIPS = 4
# N_SLOTS = 6
# N_TICKS = 15                    # short on purpose: contract tests, not rollouts
# SETTLED_TICKS = 300             # long enough for some flocks to order - see the settled fixture
# BASE_SEED = 20260902
#
#
# def build(tmp_dir, *, n_live=N_SLOTS, n_clips=N_CLIPS, n_slots=N_SLOTS,
#           n_ticks=N_TICKS, base_seed=BASE_SEED):
#     # allow_dirty=True throughout: the working tree during a test run is
#     # routinely dirty, and refusing to run is the behaviour under test in
#     # test_a_dirty_tree_is_refused_unless_allowed, not a thing to fight here.
#     return generate.build_dataset(PHYSICS, n_clips=n_clips, n_slots=n_slots,
#                                   n_live=n_live, n_ticks=n_ticks,
#                                   base_seed=base_seed, out_dir=tmp_dir,
#                                   allow_dirty=True)
#
#
# @pytest.fixture(scope="module")
# def full(tmp_path_factory):
#     """Every slot live — the shape the training split has."""
#     return build(tmp_path_factory.mktemp("full"))
#
#
# @pytest.fixture(scope="module")
# def settled(tmp_path_factory):
#     """Long enough that some flocks actually order.
#
#     At N_TICKS none of them do, so `settle_tick` is -1 everywhere and any test
#     of it agrees with itself. Measured at n_slots=6, base seed 20260902: at 15
#     and at 100 ticks every clip reports -1; at 300 the four clips report
#     285, -1, 104, 176. Only the last of those can catch a wrong answer.
#     """
#     return build(tmp_path_factory.mktemp("settled"), n_ticks=SETTLED_TICKS)
#
#
# @pytest.fixture(scope="module")
# def partial(tmp_path_factory):
#     """Four live birds in six slots — the perturbation split's shape."""
#     return build(tmp_path_factory.mktemp("partial"), n_live=4)
#
#
# def arrays(out_dir):
#     with np.load(out_dir / "clips.npz") as z:
#         return {name: z[name] for name in z.files}
#
#
# def config(out_dir):
#     return json.loads((out_dir / "config.json").read_text())
#
#
# def polarisation(velocities):
#     """Reimplemented here on purpose — see the note in test_generate.py.
#
#     Recomputing an expected value with the implementation's own function proves
#     only that the function is deterministic. This has to be an independent
#     statement of the definition or it is evidence of nothing.
#     """
#     speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
#     return np.linalg.norm((velocities / np.where(speeds == 0, 1, speeds)).mean(axis=0))
#
#
# def upcast(a):
#     """float32 off disk, float64 for arithmetic — the cast is what we measure."""
#     return a.astype(np.float64)
#
#
# # --- the files ---------------------------------------------------------------
#
# def test_the_writer_produces_the_two_files(full):
#     assert (full / "clips.npz").is_file()
#     assert (full / "config.json").is_file()
#
#
# def test_build_dataset_returns_the_directory_it_wrote(tmp_path):
#     assert build(tmp_path) == tmp_path
#
#
# # --- the shape contract, one level up ----------------------------------------
#
# def test_the_stored_shapes_carry_a_clips_axis(full):
#     a = arrays(full)
#
#     assert a["positions"].shape == (N_CLIPS, N_TICKS + 1, N_SLOTS, 2)
#     assert a["velocities"].shape == (N_CLIPS, N_TICKS + 1, N_SLOTS, 2)
#     assert a["mask"].shape == (N_CLIPS, N_TICKS + 1, N_SLOTS)
#     assert a["kind"].shape == (N_CLIPS, N_SLOTS)
#     assert a["settle_tick"].shape == (N_CLIPS,)
#
#
# def test_the_force_arrays_are_never_padded_to_match_the_states(full):
#     """The 301/300 asymmetry is the truth and belongs in the shape.
#
#     Padding forces[0] with zeros to square the arrays off fabricates a label for
#     a tick where no force has been computed — and it is invisible afterwards,
#     because zero is a perfectly ordinary force.
#     """
#     a = arrays(full)
#
#     for rule in ("separation", "alignment", "cohesion"):
#         assert a[rule].shape == (N_CLIPS, N_TICKS, N_SLOTS, 2), rule
#
#
# def test_the_stored_dtypes(full):
#     """float32 for the floats, and nothing silently widened back to float64."""
#     a = arrays(full)
#
#     for key in ("positions", "velocities", "separation", "alignment", "cohesion"):
#         assert a[key].dtype == np.float32, f"{key} is {a[key].dtype}"
#     assert a["mask"].dtype == np.bool_
#     assert a["kind"].dtype == np.int8
#     assert a["settle_tick"].dtype == np.int32
#
#
# def test_kind_is_all_birds_for_now(full):
#     assert np.array_equal(arrays(full)["kind"], np.zeros((N_CLIPS, N_SLOTS), np.int8))
#
#
# # --- config.json -------------------------------------------------------------
#
# def test_the_physics_round_trips_field_for_field(full):
#     """Values alone do not identify a dataset, but they are half of it.
#
#     Reconstructing the dataclass rather than comparing dicts also proves the
#     stored keys are the field names, so a later rename fails here rather than
#     quietly producing a config nothing can load.
#     """
#     assert Physics(**config(full)["physics"]) == PHYSICS
#
#
# def test_the_generation_parameters_are_recorded(full):
#     gen = config(full)["generation"]
#
#     assert gen["n_clips"] == N_CLIPS
#     assert gen["n_slots"] == N_SLOTS
#     assert gen["n_live"] == N_SLOTS
#     assert gen["n_ticks"] == N_TICKS
#     assert gen["base_seed"] == BASE_SEED
#
#
# def test_provenance_is_recorded(full):
#     """The other half. Values say what was asked for; the SHA says what ran.
#
#     A dataset generated by code that has since changed is not reproducible from
#     its parameters, and the failure is silent — the numbers still look fine.
#     """
#     prov = config(full)["provenance"]
#
#     assert prov["commit"], "no commit SHA recorded"
#     assert prov["commit"] == subprocess.run(
#         ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
#         check=True).stdout.strip()
#     assert "tag" in prov
#     assert isinstance(prov["dirty"], bool)
#
#
# def test_the_allow_dirty_flag_is_recorded_even_when_the_tree_is_clean(full):
#     """Provenance that can be silently wrong is decoration."""
#     assert config(full)["generation"]["allow_dirty"] is True
#
#
# def test_a_dirty_tree_is_refused_unless_allowed(tmp_path, monkeypatch):
#     """Refuse by default, and only the flag gets past it.
#
#     A dataset generated from uncommitted code records a SHA that does not
#     describe the code that ran. That is worse than no provenance, because it
#     reads as trustworthy.
#     """
#     monkeypatch.setattr(generate, "git_is_dirty", lambda: True)
#
#     with pytest.raises(RuntimeError):
#         generate.build_dataset(PHYSICS, n_clips=1, n_slots=N_SLOTS,
#                                n_live=N_SLOTS, n_ticks=2, base_seed=0,
#                                out_dir=tmp_path)
#
#     generate.build_dataset(PHYSICS, n_clips=1, n_slots=N_SLOTS, n_live=N_SLOTS,
#                            n_ticks=2, base_seed=0, out_dir=tmp_path,
#                            allow_dirty=True)
#     assert config(tmp_path)["provenance"]["dirty"] is True
#
#
# # --- the git helpers ---------------------------------------------------------
#
# def test_git_is_dirty_agrees_with_git_itself():
#     """Compared against the command, not against a remembered answer."""
#     porcelain = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
#                                text=True, check=True).stdout.strip()
#
#     assert generate.git_is_dirty() == bool(porcelain)
#
#
# def test_git_is_dirty_returns_a_real_bool():
#     """`if dirty` would be happy with the raw porcelain string too.
#
#     It is written into config.json, where a JSON string "" or " M generate.py"
#     in place of true/false would be read back by anything downstream as
#     truthy - an empty string being falsey is not something to rely on across a
#     serialisation boundary.
#     """
#     assert isinstance(generate.git_is_dirty(), bool)
#
#
# def test_provenance_describes_the_code_not_the_callers_directory(tmp_path, monkeypatch):
#     """The reason `_git` passes `-C` at generate.py's own directory.
#
#     Without it, git runs wherever the caller happened to be standing: a
#     different repo's SHA, or a crash, depending on where that is. The SHA is
#     supposed to identify the code that ran.
#     """
#     expected = generate.git_provenance()["commit"]
#     monkeypatch.chdir(tmp_path)          #not a git repository at all
#
#     assert generate.git_provenance()["commit"] == expected
#
#
# def test_provenance_commit_is_a_full_sha():
#     commit = generate.git_provenance()["commit"]
#
#     assert len(commit) == 40, commit
#     assert all(c in "0123456789abcdef" for c in commit), commit
#
#
# def test_provenance_tag_is_never_empty():
#     """`git describe --tags --always` degrades to a short SHA rather than blank.
#
#     A bare tag lookup returns nothing at an untagged HEAD, and an empty string
#     in config.json reads as "no tag" rather than "several commits past sim-v1".
#     """
#     assert generate.git_provenance()["tag"].strip()
#
#
# def test_provenance_dirty_matches_the_helper():
#     assert generate.git_provenance()["dirty"] == generate.git_is_dirty()
#
#
# # --- the seeds ---------------------------------------------------------------
#
# def test_clip_seed_reproduces_the_spawned_child_and_does_not_depend_on_n():
#     """`SeedSequence(entropy=base, spawn_key=(i,))`, not `spawn(n)[i].entropy`.
#
#     Every child of `spawn()` inherits the parent's entropy, so a seeds column
#     holding `child.entropy` writes the same number into every row and reseeding
#     from it gives one identical stream for all clips — a different stream from
#     the one that generated any of them. What identifies a child is the pair
#     (entropy, spawn_key). Verified equal for two different n, so the identity
#     does not depend on how many clips were asked for.
#     """
#     for n in (4, 9):
#         for i in range(n):
#             np.testing.assert_array_equal(
#                 generate.clip_seed(BASE_SEED, i).generate_state(4),
#                 np.random.SeedSequence(BASE_SEED).spawn(n)[i].generate_state(4),
#                 err_msg=f"clip {i} of {n}")
#
#
# def test_clip_seed_gives_every_clip_a_different_stream():
#     states = {tuple(generate.clip_seed(BASE_SEED, i).generate_state(4))
#               for i in range(N_CLIPS)}
#
#     assert len(states) == N_CLIPS, "some clips share a random stream"
#
#
# def test_a_stored_clip_regenerates_from_the_base_seed_and_its_index(full):
#     """The claim that makes the seeds column unnecessary.
#
#     If this passes, `(base_seed, index)` is enough to reconstruct any clip in
#     the dataset exactly, and `config.json` plus the array index carry both.
#     """
#     a = arrays(full)
#     index = 2
#     again = generate.generate_clip(PHYSICS, n_slots=N_SLOTS, n_live=N_SLOTS,
#                                    n_ticks=N_TICKS,
#                                    seed=generate.clip_seed(BASE_SEED, index))
#
#     for key in ("positions", "velocities", "separation", "alignment", "cohesion"):
#         np.testing.assert_array_equal(
#             a[key][index], again[key].astype(np.float32), err_msg=key)
#     assert int(a["settle_tick"][index]) == again["settle_tick"]
#
#
# def test_the_clips_are_not_all_the_same_flock(full):
#     """Cheap, and it catches seeding every clip from the base seed itself."""
#     positions = arrays(full)["positions"]
#
#     for i in range(1, N_CLIPS):
#         assert not np.array_equal(positions[0], positions[i]), f"clip {i} == clip 0"
#
#
# # --- the invariants, restated against what is actually on disk ---------------
#
# def test_forces_reconstruct_velocities_after_the_float32_round_trip(full):
#     """Tick alignment, surviving the cast — the one that matters most here.
#
#     Exactness is gone by definition once the arrays are float32, so this is a
#     tolerance. The measured worst case for this round trip is 2.9e-07, about
#     2.4x float32 epsilon (tests/test_step.py), so a failure at 1e-6 means
#     something is misaligned rather than that the bound is tight.
#
#     Checked against `velocities[t]` and `velocities[t+1]` on purpose: an
#     internally consistent but shifted recording passes every check that looks
#     only at the forces.
#     """
#     a = arrays(full)
#     v = upcast(a["velocities"])
#     total = upcast(a["separation"]) + upcast(a["alignment"]) + upcast(a["cohesion"])
#
#     worst = 0.0
#     for c in range(N_CLIPS):
#         for t in range(N_TICKS):
#             for i in range(N_SLOTS):
#                 expected = velocity_capper(v[c, t, i] + total[c, t, i] * PHYSICS.dt,
#                                            PHYSICS.max_speed, PHYSICS.min_speed)
#                 worst = max(worst, float(np.abs(expected - v[c, t + 1, i]).max()))
#
#     assert worst < 1e-6, f"round trip lost {worst:.3e} — more than the cast alone"
#
#
# def test_mask_and_nan_are_exact_complements_on_disk(partial):
#     """The checksum a mask alone cannot provide.
#
#     With zero-filled dead slots there is no way to tell an empty slot from a
#     bird at the origin, and 0.0 is a legal velocity component too. NaN makes the
#     two representations checkable against each other, which is why both are
#     stored.
#     """
#     a = arrays(partial)
#     dead = ~a["mask"]
#
#     assert dead.any(), "premise failed: nothing was masked off"
#     np.testing.assert_array_equal(np.isnan(a["positions"][..., 0]), dead)
#     np.testing.assert_array_equal(np.isnan(a["velocities"][..., 0]), dead)
#     for rule in ("separation", "alignment", "cohesion"):
#         np.testing.assert_array_equal(np.isnan(a[rule][..., 0]), dead[:, :-1], rule)
#
#
# def test_no_nan_in_live_slots(partial):
#     """NaN shouts on comparisons and whispers on aggregates.
#
#     Every check in this file is written over live slots explicitly rather than
#     trusting NaN to surface a mistake, because `mean()` propagates it in
#     silence. This test is the one that makes that safe to rely on.
#     """
#     a = arrays(partial)
#     live = a["mask"]
#
#     assert not np.isnan(a["positions"][live]).any()
#     assert not np.isnan(a["velocities"][live]).any()
#     for rule in ("separation", "alignment", "cohesion"):
#         assert not np.isnan(a[rule][live[:, :-1]]).any(), rule
#
#
# def test_positions_and_speeds_are_legal_including_tick_0(full):
#     """Tick 0 is the one the simulator never produced, so it is the one to check.
#
#     The upper bound on position is closed rather than half-open here on purpose:
#     a float64 coordinate a hair under `bounds` can round *up* to exactly
#     `bounds` in float32. That is a property of the cast, not a wrapping bug, and
#     asserting `< bounds` would make this test fail at random.
#     """
#     a = arrays(full)
#     positions = upcast(a["positions"])[a["mask"]]
#     speeds = np.linalg.norm(upcast(a["velocities"])[a["mask"]], axis=-1)
#
#     assert positions.min() >= 0.0
#     assert positions.max() <= PHYSICS.bounds
#     assert speeds.min() >= PHYSICS.min_speed - 1e-5, f"slowest {speeds.min()}"
#     assert speeds.max() <= PHYSICS.max_speed + 1e-5, f"fastest {speeds.max()}"
#
#
# def test_a_clip_with_no_ticks_still_has_a_legal_shape(tmp_path):
#     """n_ticks=0 takes its own branch in the writer and nothing exercised it.
#
#     One state, no forces. The degenerate case is worth pinning because the
#     branch that builds empty force arrays is the one place the 301/300
#     asymmetry is written out by hand rather than falling out of the loop.
#     """
#     out = build(tmp_path, n_clips=1, n_ticks=0)
#     a = arrays(out)
#
#     assert a["positions"].shape == (1, 1, N_SLOTS, 2)
#     for rule in ("separation", "alignment", "cohesion"):
#         assert a[rule].shape == (1, 0, N_SLOTS, 2), rule
#     assert int(a["settle_tick"][0]) in (-1, 0)
#
#
# def test_thirty_clips_are_all_legal(tmp_path):
#     """Breadth rather than depth: one seed proves nothing about the others.
#
#     Every measured claim in this project that turned out wrong was generalised
#     from a single seed. This asserts nothing about the *values* - those are what
#     the summary report is for - only that no clip in a larger sample violates
#     the world's own rules.
#     """
#     out = build(tmp_path, n_clips=30, n_ticks=25)
#     a = arrays(out)
#     live = a["mask"]
#
#     assert not np.isnan(a["positions"][live]).any()
#     positions = upcast(a["positions"])[live]
#     speeds = np.linalg.norm(upcast(a["velocities"])[live], axis=-1)
#
#     assert positions.min() >= 0.0 and positions.max() <= PHYSICS.bounds
#     assert speeds.min() >= PHYSICS.min_speed - 1e-5
#     assert speeds.max() <= PHYSICS.max_speed + 1e-5
#     assert np.all(a["settle_tick"] >= -1) and np.all(a["settle_tick"] <= 25)
#     np.testing.assert_array_equal(np.isnan(a["positions"][..., 0]), ~live)
#
#
# def test_settle_tick_matches_a_recomputation_from_the_stored_velocities(settled):
#     """Recomputed from the file, not from the generator's own bookkeeping.
#
#     Deliberately on the long fixture. At the short one no flock reaches
#     polarisation 0.99, so both sides of the comparison are -1 and an
#     implementation that never looks at the velocities passes - which is what
#     this test did until the premise assertion below was added.
#     """
#     a = arrays(settled)
#
#     computed = []
#     for c in range(N_CLIPS):
#         live = a["mask"][c, 0]
#         expected = -1
#         for t in range(SETTLED_TICKS + 1):
#             if polarisation(upcast(a["velocities"][c, t][live])) > 0.99:
#                 expected = t
#                 break
#         computed.append(expected)
#         assert int(a["settle_tick"][c]) == expected, f"clip {c}"
#
#     assert any(t != -1 for t in computed), (
#         "premise failed: no clip settled, so this test compared -1 with -1 and "
#         "would pass against a hardcoded answer")
