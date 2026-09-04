"""calculate_cohesion: steer toward the local centre of mass.

Two things make this the most dangerous of the three rules to get wrong.

The centre of mass of a set of positions is meaningless on a torus - the mean
of x=1 and x=99 is 50, the far side of the world - so the rule must average
*offsets*, and several tests here exist only to prove that it does.

And the force is a spring: magnitude proportional to displacement from the
centre, rather than the 1/d law of separation or the flat average of
alignment. Normalising it to a unit vector would be the natural thing to write
and would silently destroy the label, so that is pinned too.
"""

import warnings

import numpy as np
import pytest

from boids import calculate_cohesion
from world import BOUNDS, COHESION_RADIUS


def test_cohesion_pulls_toward_a_lone_neighbour():
    """The sign test.

    Separation and cohesion differ by the order of two arguments to get_torus
    and nothing else. Swapping them inverts the physics, raises nothing, and
    produces a force vector of exactly the right magnitude.
    """
    positions = np.array([[50.0, 50.0],
                          [60.0, 50.0]])   # neighbour 10 units in +x

    force = calculate_cohesion(0, positions, COHESION_RADIUS, BOUNDS)

    np.testing.assert_allclose(force, [0.25, 0.0])   # 10 / 40
    assert force[0] > 0, (
        "cohesion must pull toward the neighbour; a negative x here means the "
        "arguments to get_torus are the wrong way round and this is separation"
    )


def test_cohesion_averages_offsets_not_positions():
    """The trap this rule is famous for.

    Bird 0 sits at x=0 with neighbours at x=2 and x=99 - which on a torus are
    2 units to its right and 1 unit to its left. The true mean offset is
    [0.5, 0], a gentle pull in +x.

    Average the raw positions instead and you get (2 + 99) / 2 = 50.5, i.e. a
    centre of mass on the far side of the world, and a force of about
    [-1.24, 0]: the opposite direction, a hundred times too strong, and
    entirely plausible-looking.
    """
    positions = np.array([[0.0, 50.0],
                          [2.0, 50.0],
                          [99.0, 50.0]])

    force = calculate_cohesion(0, positions, COHESION_RADIUS, BOUNDS)

    np.testing.assert_allclose(force, [0.0125, 0.0])
    assert force[0] > 0, "a negative force here means positions were averaged"


def test_cohesion_neighbour_across_the_seam_counts():
    """x=2 and x=98 are 4 units apart, and the pull is leftward through 0."""
    positions = np.array([[2.0, 50.0],
                          [98.0, 50.0]])

    force = calculate_cohesion(0, positions, COHESION_RADIUS, BOUNDS)

    np.testing.assert_allclose(force, [-0.1, 0.0])   # -4 / 40


def test_cohesion_switches_off_at_the_centre_of_mass():
    """A bird already at the centre of its flock must feel nothing.

    This is the property that normalising the mean to a unit vector destroys:
    under normalisation a bird at the exact centre gets [0,0] but a bird a
    thousandth of a unit away gets a full-magnitude force.
    """
    positions = np.array([[50.0, 50.0],
                          [45.0, 50.0],
                          [55.0, 50.0]])

    force = calculate_cohesion(0, positions, COHESION_RADIUS, BOUNDS)

    np.testing.assert_allclose(force, [0.0, 0.0], atol=1e-12)


def test_cohesion_is_small_just_off_the_centre():
    """The other half of the previous test, and the one with teeth.

    Bird 0 is 0.02 units off its flock's centre of mass. The force must be
    correspondingly tiny. An implementation that returns mean / norm(mean)
    returns magnitude 1.0 here - two thousand times too large - while still
    passing every test that only checks direction.
    """
    positions = np.array([[50.0, 50.0],
                          [45.0, 50.0],
                          [55.04, 50.0]])

    force = calculate_cohesion(0, positions, COHESION_RADIUS, BOUNDS)

    np.testing.assert_allclose(force, [0.0005, 0.0])
    assert np.linalg.norm(force) < 0.001, (
        "a magnitude near 1.0 means the mean offset was normalised, which "
        "makes the force discontinuous at the centre of mass"
    )


def test_cohesion_grows_linearly_with_displacement():
    """Hooke's law: magnitude proportional to distance from the centre.

    Three worlds, one neighbour each, at 4, 20 and 39 units. Any rule that is
    normalised, or that inherits separation's 1/d falloff, breaks this.
    """
    forces = [
        calculate_cohesion(
            0,
            np.array([[50.0, 50.0], [50.0 + d, 50.0]]),
            COHESION_RADIUS,
            BOUNDS,
        )
        for d in (4.0, 20.0, 39.0)
    ]
    magnitudes = [np.linalg.norm(f) for f in forces]

    assert magnitudes == pytest.approx([0.1, 0.5, 0.975])
    assert magnitudes[0] < magnitudes[1] < magnitudes[2]


def test_cohesion_magnitude_never_reaches_one():
    """Bounded by construction, which is why this rule needs no force cap.

    Every offset in the sum has norm below cohesion_radius, so their mean does
    too, so mean / cohesion_radius is inside the unit disc. Thirty birds
    scattered at random must all obey it.
    """
    rng = np.random.default_rng(0)
    positions = rng.uniform(0.0, BOUNDS, size=(30, 2))

    magnitudes = np.array([
        np.linalg.norm(calculate_cohesion(i, positions, COHESION_RADIUS, BOUNDS))
        for i in range(len(positions))
    ])

    assert magnitudes.max() < 1.0
    # premise guard: a function returning zeros would satisfy the bound
    assert magnitudes.max() > 0.0


def test_cohesion_excludes_neighbours_outside_the_radius():
    """A third bird 45 units away must not enter the average at all.

    If it did, the answer would differ from the two-bird case below - and the
    divisor would be 2 rather than 1, which is the quieter half of the bug.
    """
    with_far_bird = np.array([[50.0, 50.0],
                              [60.0, 50.0],
                              [95.0, 50.0]])   # 45 units away, outside 40

    force = calculate_cohesion(0, with_far_bird, COHESION_RADIUS, BOUNDS)

    np.testing.assert_allclose(force, [0.25, 0.0])


def test_cohesion_counts_a_coincident_neighbour():
    """Unlike separation, cohesion has no `0 < dist` guard, deliberately.

    Nothing divides by dist here, so a bird sitting exactly on top of another
    cannot produce a NaN. It contributes a zero offset and still counts toward
    the divisor: mean of [0,0] and [10,0] is [5,0], not [10,0].
    """
    positions = np.array([[50.0, 50.0],
                          [50.0, 50.0],
                          [60.0, 50.0]])

    force = calculate_cohesion(0, positions, COHESION_RADIUS, BOUNDS)

    assert not np.isnan(force).any()
    np.testing.assert_allclose(force, [0.125, 0.0])   # 5 / 40


def test_cohesion_isolated_bird_feels_nothing_and_divides_by_nothing():
    """No neighbours means no force - and no 0/0 on the way there.

    numpy answers np.zeros(2) / 0 with [nan, nan] and a RuntimeWarning rather
    than an exception, so a guard placed after the division still returns the
    right answer while quietly emitting a warning per isolated bird per tick.
    Under np.seterr(all='raise') - which the dataset generator should use - it
    becomes a hard crash mid-run instead.
    """
    positions = np.array([[10.0, 10.0],
                          [50.0, 50.0]])   # torus distance 56.6, outside 40

    with warnings.catch_warnings():
        warnings.simplefilter("error")   # any RuntimeWarning fails this test
        force = calculate_cohesion(0, positions, COHESION_RADIUS, BOUNDS)

    assert not np.isnan(force).any()
    np.testing.assert_array_equal(force, [0.0, 0.0])


def test_cohesion_single_bird_world():
    """One bird, no neighbours to skip, nothing to divide by."""
    positions = np.array([[10.0, 10.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        force = calculate_cohesion(0, positions, COHESION_RADIUS, BOUNDS)

    assert not np.isnan(force).any()
    np.testing.assert_array_equal(force, [0.0, 0.0])
