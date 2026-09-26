from dataclasses import replace

import numpy as np
import pytest

import boids
from boids import (Physics, calculate_alignment, calculate_cohesion, calculate_separation,
                   step, velocity_capper)
from world import PHYSICS, PHYSICS_CRUISING, random_flock

if not hasattr(boids, "force_capper"):
    pytest.skip("boids.force_capper does not exist yet, this suite is its "
                "specification", allow_module_level=True)

from boids import force_capper

CAP = 0.3


# --- the constants ---------------------------------------------------------------------

def test_the_shipped_softness_and_cap():
    """The pair measured to balance the rules: with them separation and
    cohesion carry about 12% and 10% of the forming-phase motion, against 2%
    each without, and about half of all flocks still settle within 300 ticks."""
    shipped = Physics()
    assert shipped.edge_softness == 0.25
    assert shipped.max_force == 0.3


@pytest.mark.parametrize("softness", [-0.1, 1.5])
def test_a_softness_outside_zero_to_one_is_refused(softness):
    with pytest.raises(ValueError):
        Physics(edge_softness=softness)


@pytest.mark.parametrize("softness", [0.0, 1.0])
def test_the_ends_of_the_softness_range_are_allowed(softness):
    assert Physics(edge_softness=softness).edge_softness == softness


@pytest.mark.parametrize("cap", [0.0, -0.3])
def test_a_cap_that_is_not_positive_is_refused(cap):
    with pytest.raises(ValueError):
        Physics(max_force=cap)


def test_no_cap_is_allowed():
    assert Physics(max_force=None).max_force is None


# --- force_capper ----------------------------------------------------------------------

def test_a_short_force_passes_through_untouched():
    np.testing.assert_array_equal(force_capper(np.array([0.1, -0.2]), CAP), [0.1, -0.2])


def test_a_force_exactly_at_the_cap_is_untouched():
    f = np.array([0.18, 0.24])        # length 0.3
    np.testing.assert_array_equal(force_capper(f, CAP), f)


def test_a_long_force_is_shrunk_to_the_cap():
    got = force_capper(np.array([3.0, -4.0]), CAP)
    assert np.linalg.norm(got) == pytest.approx(CAP, abs=1e-12)


def test_shrinking_keeps_the_direction():
    f = np.array([-2.0, 7.0])
    got = force_capper(f, CAP)
    assert got @ f > 0
    assert abs(got[0] * f[1] - got[1] * f[0]) < 1e-12


def test_the_cap_limits_the_length_not_each_axis():
    """Each axis is under the cap, the length is not. Clipping axes separately
    would let a diagonal push through at up to 1.41 times the cap."""
    got = force_capper(np.array([0.25, 0.25]), CAP)
    assert np.linalg.norm(got) == pytest.approx(CAP, abs=1e-12)


def test_no_cap_means_no_cap():
    np.testing.assert_array_equal(force_capper(np.array([30.0, -40.0]), None), [30.0, -40.0])


def test_a_zero_force_stays_zero():
    got = force_capper(np.zeros(2), CAP)
    assert np.all(got == 0.0) and np.all(np.isfinite(got))


def test_the_capper_does_not_change_what_it_is_handed():
    f = np.array([3.0, -4.0])
    force_capper(f, CAP)
    np.testing.assert_array_equal(f, [3.0, -4.0])


# --- the cap inside step ------------------------------------------------------------------

def fast_scattered_flock(seed):
    """Birds in random directions at up to max speed: alignment's push is
    large here, which is exactly what the cap exists for."""
    return random_flock(seed=seed, n_birds=12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_recorded_force_is_within_the_cap(seed):
    positions, velocities = fast_scattered_flock(seed)
    _, _, forces = step(positions, velocities, PHYSICS_CRUISING)
    for rule, f in forces.items():
        assert np.linalg.norm(f, axis=1).max() <= CAP + 1e-12, f"{rule} over the cap"
    at_cap = np.isclose(np.linalg.norm(forces["alignment"], axis=1), CAP, atol=1e-12)
    assert at_cap.any(), "premise: no alignment force reached the cap, so none was capped"


def test_the_recorded_forces_rebuild_the_velocity_update_under_the_cap():
    """The label invariant, with the cap firing: the recorded forces are the
    capped ones, the ones that moved the bird. Recording the raw force while
    applying the capped one would break this and nothing else."""
    positions, velocities = fast_scattered_flock(3)
    ph = PHYSICS_CRUISING
    _, new_velocities, forces = step(positions, velocities, ph)
    raw_alignment = np.array([ph.w_ali * calculate_alignment(i, positions, velocities, ph.alignment_radius,
                                                             ph.bounds, softness=ph.edge_softness)
                              for i in range(12)])
    assert (np.linalg.norm(raw_alignment, axis=1) > CAP).any(), "premise: the cap must fire"
    for i in range(12):
        total = forces["separation"][i] + forces["alignment"][i] + forces["cohesion"][i]
        np.testing.assert_allclose(new_velocities[i],
                                   velocity_capper(velocities[i] + total * ph.dt, ph.max_speed, ph.min_speed),
                                   rtol=1e-12, atol=1e-15)


def test_each_rule_is_capped_on_its_own_not_the_total():
    """Alignment and cohesion each under the cap, their sum over it: nothing
    may be shrunk, and the world must move exactly as it would uncapped."""
    positions = np.array([[50.0, 50.0], [50.0, 75.0]])     # 25 apart
    velocities = np.array([[1.0, 0.0], [1.0, 0.25]])
    ph = replace(PHYSICS, max_force=CAP)
    _, v_capped, f_capped = step(positions, velocities, ph)
    _, v_free, f_free = step(positions, velocities, replace(ph, max_force=None))

    total = f_free["separation"][0] + f_free["alignment"][0] + f_free["cohesion"][0]
    for rule in f_free:
        assert np.linalg.norm(f_free[rule], axis=1).max() < CAP, f"premise: {rule} over the cap"
    assert np.linalg.norm(total) > CAP, "premise: the sum must be over the cap"
    np.testing.assert_array_equal(v_capped, v_free)
    for rule in f_free:
        np.testing.assert_array_equal(f_capped[rule], f_free[rule])


def test_the_cap_applies_after_the_weight():
    """Raw alignment 0.5, weight 0.5: the weighted push is 0.25, under the
    cap, so it is recorded whole. Capping first gives 0.3 * 0.5 = 0.15."""
    positions = np.array([[50.0, 50.0], [50.0, 55.0]])
    velocities = np.array([[1.0, 0.0], [1.0, 0.5]])
    ph = replace(PHYSICS, w_ali=0.5, max_force=CAP)
    _, _, forces = step(positions, velocities, ph)
    np.testing.assert_allclose(forces["alignment"][0], [0.0, 0.25], rtol=1e-12)


def test_step_reads_max_force():
    positions, velocities = fast_scattered_flock(4)
    _, _, a = step(positions, velocities, replace(PHYSICS, max_force=0.3))
    _, _, b = step(positions, velocities, replace(PHYSICS, max_force=0.5))
    assert not np.array_equal(a["alignment"], b["alignment"])


def test_step_reads_edge_softness():
    positions, velocities = fast_scattered_flock(5)
    _, _, a = step(positions, velocities, replace(PHYSICS, edge_softness=0.25))
    _, _, b = step(positions, velocities, replace(PHYSICS, edge_softness=0.0))
    assert any(not np.array_equal(a[r], b[r]) for r in a)


def test_step_softens_all_three_rules():
    """A neighbour in each rule's fade zone: 17 units (separation, 15-20),
    27 (alignment, 22.5-30), 37 (cohesion, 30-40). Each recorded force must be
    its rule at the physics' softness, and must differ from the hard edge."""
    here = np.array([50.0, 50.0])
    positions = np.array([here, here + [17.0, 0.0], here + [0.0, 27.0], here + [-37.0, 0.0]])
    velocities = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, -0.2], [1.1, 0.3]])
    ph = replace(PHYSICS, edge_softness=0.25, max_force=None)
    _, _, forces = step(positions, velocities, ph)

    soft = {"separation": ph.w_sep * calculate_separation(0, positions, ph.separation_radius, ph.bounds,
                                                          ph.sep_force_cap, softness=0.25),
            "alignment": ph.w_ali * calculate_alignment(0, positions, velocities, ph.alignment_radius,
                                                        ph.bounds, softness=0.25),
            "cohesion": ph.w_coh * calculate_cohesion(0, positions, ph.cohesion_radius, ph.bounds,
                                                      softness=0.25)}
    hard = {"separation": ph.w_sep * calculate_separation(0, positions, ph.separation_radius, ph.bounds,
                                                          ph.sep_force_cap),
            "alignment": ph.w_ali * calculate_alignment(0, positions, velocities, ph.alignment_radius, ph.bounds),
            "cohesion": ph.w_coh * calculate_cohesion(0, positions, ph.cohesion_radius, ph.bounds)}
    for rule in soft:
        assert not np.allclose(soft[rule], hard[rule]), f"premise: {rule} is not in its fade zone"
        np.testing.assert_allclose(forces[rule][0], soft[rule], rtol=1e-12,
                                   err_msg=f"{rule} was not given the softness")


def test_the_shipped_world_still_flocks():
    """The cap must trim the early chaos, not stop flocks forming. These six
    flocks reach a mean polarisation of 0.80 after 250 ticks; with alignment's
    weight lowered to 0.1 and no cap they reach 0.57, which is the failure this
    guards."""
    rng = np.random.default_rng(20260926)
    ph = PHYSICS_CRUISING
    final = []
    for _ in range(6):
        p = rng.uniform(0, ph.bounds, (12, 2))
        h = rng.uniform(0, 2 * np.pi, 12)
        s = rng.uniform(ph.min_speed, ph.max_speed, 12)
        v = np.stack([s * np.cos(h), s * np.sin(h)], 1)
        for _ in range(250):
            p, v, _ = step(p, v, ph)
        u = v / np.linalg.norm(v, axis=1, keepdims=True)
        final.append(np.linalg.norm(u.mean(0)))
    assert np.mean(final) > FLOCK_BAR, f"mean polarisation after 250 ticks {np.mean(final):.3f}"


FLOCK_BAR = 0.7
