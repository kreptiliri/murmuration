# ==========================================================================
# THE WHOLE SUITE IN ONE FILE - reference copy, entirely commented out.
#
# This is not run by pytest. Every line below is a comment, and the leading
# underscore keeps the filename from matching pytest's test_*.py discovery
# pattern, so nothing here is collected or executed twice.
#
# It exists to be read end-to-end. The live suite is the six files this was
# assembled from; if you change a test, change it there - nothing keeps this
# copy in sync, and a stale reference copy is worse than none.
#
# Assembled from, in order:
#   world.py, test_geometry.py, test_separation.py, test_alignment.py,
#   test_cohesion.py, test_step.py, test_rollout.py
# ==========================================================================


# ==========================================================================
# world.py
# ==========================================================================
# """The world every test in this suite runs in.
#
# Shared so that the five test modules cannot drift into disagreeing about the
# size of the world or how fast a bird may fly. Tests that need to vary one
# constant use dataclasses.replace(PHYSICS, thing=...), which changes that field
# and leaves the other nine exactly as they are here.
# """
#
# import numpy as np
#
# from boids import Physics
#
# # a 100x100 torus
# BOUNDS = 100.0
# SEPARATION_RADIUS = 20.0
# ALIGNMENT_RADIUS = 35.0
# COHESION_RADIUS = 40.0
# MAX_SPEED = 5.0
# DT = 0.1
#
# PHYSICS = Physics(
#     dt=DT,
#     bounds=BOUNDS,
#     max_speed=MAX_SPEED,
#     separation_radius=SEPARATION_RADIUS,
#     alignment_radius=ALIGNMENT_RADIUS,
#     cohesion_radius=COHESION_RADIUS,
# )
#
#
# def random_flock(seed, n_birds):
#     """A deterministic starting world. Same seed, same flock, always."""
#     rng = np.random.default_rng(seed)
#     positions = rng.uniform(0.0, BOUNDS, size=(n_birds, 2))
#     velocities = rng.uniform(-MAX_SPEED, MAX_SPEED, size=(n_birds, 2))
#     return positions, velocities


# ==========================================================================
# test_geometry.py
# ==========================================================================
# """The primitives every rule is built on: torus distance, the speed cap,
# and the Physics config object's own validation."""
#
# from dataclasses import FrozenInstanceError
#
# import numpy as np
# import pytest
#
# from boids import Physics, get_torus, velocity_capper
# from world import BOUNDS, MAX_SPEED
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


# ==========================================================================
# test_separation.py
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
# test_alignment.py
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
#     A neighbour at d = 5 and the same neighbour at d = 30 must produce an
#     identical force. This is what catches alignment written by copy-pasting
#     calculate_separation and forgetting to strip the distance weighting.
#     """
#     velocities = np.array([[1.0, 0.0],
#                            [0.0, 1.0]])
#     near = np.array([[50.0, 50.0], [55.0, 50.0]])   # d = 5
#     far = np.array([[50.0, 50.0], [80.0, 50.0]])    # d = 30, still inside 35
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
# test_cohesion.py
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
# test_step.py
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
# from boids import calculate_separation, step, velocity_capper
# from world import BOUNDS, DT, MAX_SPEED, PHYSICS, SEPARATION_RADIUS
#
#
# def _reconstruct_velocity(velocities, forces, boid_id, dt, max_speed):
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
#     return velocity_capper(velocities[boid_id] + total * dt, max_speed)
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
#             _reconstruct_velocity(velocities, forces, i, DT, MAX_SPEED),
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
#     max_speed = 0.5  # low enough that every bird below exceeds it
#     positions = np.array([[10.0, 10.0],
#                           [15.0, 10.0],
#                           [80.0, 80.0]])
#     velocities = np.array([[1.0, 0.0],
#                            [-1.0, 0.0],
#                            [0.5, 0.5]])
#
#     _, new_velocities, forces = step(
#         positions, velocities, replace(PHYSICS, max_speed=max_speed)
#     )
#
#     # Guard the test's own premise: if the cap never fires this proves nothing.
#     uncapped = np.array([
#         velocities[i] + (forces["separation"][i]
#                          + forces["alignment"][i]
#                          + forces["cohesion"][i]) * DT
#         for i in range(len(positions))
#     ])
#     assert np.any(np.linalg.norm(uncapped, axis=1) > max_speed), (
#         "premise failed: the cap never fired, so this test verifies nothing"
#     )
#
#     for i in range(len(positions)):
#         np.testing.assert_allclose(
#             _reconstruct_velocity(velocities, forces, i, DT, max_speed),
#             new_velocities[i],
#             err_msg=f"reconstruction broke for bird {i} once the cap fired",
#         )
#         assert np.linalg.norm(new_velocities[i]) <= max_speed + 1e-12
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
#     """The same pair of birds, neighbours under one radius and not the other."""
#     positions = np.array([[50.0, 50.0],
#                           [80.0, 50.0]])   # d = 30
#     velocities = np.array([[1.0, 0.0],
#                            [0.0, 1.0]])
#
#     _, _, wide = step(positions, velocities, replace(PHYSICS, alignment_radius=35.0))
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


# ==========================================================================
# test_rollout.py
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
# from boids import step, velocity_capper
# from world import BOUNDS, DT, MAX_SPEED, PHYSICS, random_flock
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
#                 velocity_capper(uncapped[i], physics.max_speed),
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
