
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Physics:
    dt: float = 0.1
    bounds: float = 100.0
    max_speed: float = 5.0
    min_speed: float = 1.0
    separation_radius: float = 20.0
    alignment_radius: float = 30.0   
    cohesion_radius: float = 40.0
    sep_force_cap: float = 1.0
    edge_softness: float = 0.25
    max_force: float | None = 0.3
    w_sep: float = 1.5
    w_ali: float = 1.0
    w_coh: float = 0.3

    def __post_init__(self):
        if self.bounds <= 0 or self.dt <= 0 or self.max_speed <= 0 or self.min_speed < 0:
            raise ValueError("bounds, dt, max_speed and min_speed must all be positive")
        if self.min_speed > self.max_speed:
            raise ValueError("min_speed cannot be greater than max_speed")
        #Minimum-image caps any axis-displacement at bounds/2, 
        #so neighbours beyond that distance are unreachable; a radius ≥ bounds/2 makes every bird a neighbour of every other, 
        #and the radius stops selecting anything
        for name in ("separation_radius", "alignment_radius", "cohesion_radius"):
            radius = getattr(self, name)
            if not 0 < radius < self.bounds / 2:
                raise ValueError(
                    f"{name}={radius} must be in (0, bounds/2) "
                    f"= (0, {self.bounds / 2}); beyond that the minimum-image "
                    "convention no longer distinguishes near from far"
                )
        if not 0 <= self.edge_softness <= 1:
            raise ValueError("softness must be in [0, 1]!")
        if self.max_force is not None and self.max_force <= 0:
            raise ValueError("max_force must either be positive or None for no cap")

def get_torus(pos_a,pos_b,bounds):
    delta = pos_a - pos_b
    
    delta = (delta + bounds/2) % bounds - bounds/2
    return delta

def edge_weight(dist, radius, softness):
    if dist >= radius:
        return 0.0
    if softness == 0:
        return 1.0
    return min(1.0, (radius - dist) / (softness * radius))

def force_capper(force, max_force):
    if max_force is None:
        return force
    length = np.linalg.norm(force)
    if length > max_force:
        return force/ length * max_force #vector back to cap
    return force

def calculate_separation(boid_id, positions, separation_radius,bounds, sep_force_cap=1.0, softness=0.0):
    my_pos = positions[boid_id]
    total_force = np.zeros(2)
    neighbor_count = 0
    total_weight = 0
    for i,neighbor_pos in enumerate(positions):
        if i == boid_id:
             continue
        offset = get_torus(my_pos,neighbor_pos,bounds) 
        dist = np.linalg.norm(offset)
        weight = edge_weight(dist, separation_radius, softness)
        if 0 < dist < separation_radius:
            #direction vector would be pointing directly away from neighbor
            direction = offset / dist
            # clamp the magnitude expression only dist stays not modified so 'direction' remains a true unit vector for labeling
            force_magnitude = min(1.0 / dist, sep_force_cap) #max is mathematically equivalent to 1/d law as well.
            total_force += direction * force_magnitude * weight
            total_weight += weight
            neighbor_count += 1 
    if neighbor_count == 0:
            return np.array([0.0, 0.0])      
    return total_force / max(1.0, total_weight)

def calculate_alignment(boid_id, positions, velocities, alignment_radius, bounds, softness=0.0):
    my_pos = positions[boid_id]
    my_velocity = velocities[boid_id]
    total_force = np.zeros(2)
    total_weight = 0
    neighbor_count = 0
    for i, neighbor_pos in enumerate(positions):
        if i == boid_id:
            continue
        offset = get_torus(my_pos,neighbor_pos,bounds)
        dist = np.linalg.norm(offset)
        weight = edge_weight(dist, alignment_radius, softness)
        if dist < alignment_radius: #deliberately left out >0 since a bird on top of another is harmless, unlike /dist in sep
            total_force += velocities[i] * weight
            neighbor_count += 1
            total_weight += weight
    if neighbor_count == 0:
        return np.array([0.0, 0.0])      
    return total_force / max(1.0, total_weight) - min(1.0, total_weight) * my_velocity #otherwise mean = my_velo; force = 0 (body moving in one direction) with min update

def calculate_cohesion(boid_id, positions, cohesion_radius, bounds, softness=0.0):
    my_pos = positions[boid_id]
    total_force = np.zeros(2)
    neighbor_count = 0
    total_weight = 0
    for i, neighbor_pos in enumerate(positions):
        if i == boid_id:
            continue
        offset = get_torus(neighbor_pos, my_pos, bounds) #arguments swapped => vector pointing me from me to my neighbor
        dist = np.linalg.norm(offset)
        weight = edge_weight(dist, cohesion_radius, softness)
        if dist < cohesion_radius:
            total_force += offset * weight
            total_weight += weight
            neighbor_count += 1
    if neighbor_count == 0: 
        return np.array([0.0, 0.0])
    mean = total_force / max(total_weight, 1.0)
    return mean / cohesion_radius #smooth 0->1 vector, spring dynamics(no 1/d regime)

def velocity_capper(old_velocity, max_speed, min_speed=0.0):
    speed = np.linalg.norm(old_velocity)
    if speed > max_speed:
        return (old_velocity/speed) * max_speed
    if 0 < speed < min_speed:
        return (old_velocity/speed) * min_speed
    #next tick gives nonzero velo
    return old_velocity
def step(positions, velocities, physics):
    #Advance the world one tick.

    #positions and velocities are variables, physics is a config
    
    dt = physics.dt
    bounds = physics.bounds
    max_speed = physics.max_speed
    min_speed = physics.min_speed
    softness = physics.edge_softness
    max_force = physics.max_force
    N = len(positions) 
    sep_forces = np.zeros((N, 2))
    ali_forces = np.zeros((N, 2))
    coh_forces = np.zeros((N,2))
    for i in range(N): #one pass per bird, all rules read the same unmutated state
        sep_forces[i] = calculate_separation(i, positions, physics.separation_radius, bounds, physics.sep_force_cap, softness)
        ali_forces[i] = calculate_alignment(i, positions, velocities, physics.alignment_radius, bounds, softness)
        coh_forces[i] = calculate_cohesion(i, positions ,physics.cohesion_radius, bounds, softness)
    forces = {
    "separation": sep_forces * physics.w_sep,
    "alignment": ali_forces * physics.w_ali,
    "cohesion": coh_forces * physics.w_coh,}
    for rule in forces:
        for i in range(N):
            forces[rule][i] = force_capper(forces[rule][i], max_force)
    new_velocities = np.zeros_like(velocities, dtype=float)
    new_positions = np.zeros_like(positions, dtype=float)
    for i in range(N):
        total_force = (
                 forces["separation"][i]
               + forces["alignment"][i]
               + forces["cohesion"][i])

        v_raw = velocities[i] + (total_force * dt) #uncapped
        # FORCES ARE RECORDED AFTER FORCE CAP, before speed cap !!!!!!!!!!!!!!!!!!!!!!!!
        new_velocities[i] = velocity_capper(v_raw, max_speed, min_speed)
        new_positions[i] = positions[i] + new_velocities[i] * dt

    new_positions %= bounds
    return new_positions, new_velocities, forces



if __name__ == "__main__":
    physics = Physics()

    test_positions = np.array([
        [10.0, 10.0],
        [15.0, 10.0],
        [80.0, 80.0],
    ])

    test_velocities = np.array([
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.5, 0.5],
    ])

    next_pos, next_vel, recorded_forces = step(test_positions, test_velocities, physics)
    print(physics)
    print(next_pos)
