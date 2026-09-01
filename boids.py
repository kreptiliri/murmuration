
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Physics:
    dt: float = 0.1
    bounds: float = 100.0
    max_speed: float = 5.0
    min_speed: float = 2.0 #i am not sure yet
    separation_radius: float = 20.0
    alignment_radius: float = 30.0   #a placeholder for future use
    cohesion_radius: float = 40.0
    sep_force_cap: float = 1.0
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


def get_torus(pos_a,pos_b,bounds):
    delta = pos_a - pos_b
    #confusing, it looks like a congruence to b over bound
    delta = (delta + bounds/2) % bounds - bounds/2
    return delta

        

def calculate_separation(boid_id, positions, separation_radius,bounds, sep_force_cap=1.0):
    my_pos = positions[boid_id]
    total_force = np.zeros(2)
    neighbor_count = 0
    for i,neighbor_pos in enumerate(positions):
        if i == boid_id:
             continue
        offset = get_torus(my_pos,neighbor_pos,bounds) #what, nvm this should be looped
        dist = np.linalg.norm(offset)
        if 0 < dist < separation_radius:
            #direction vector would be pointing directly away from neighbor
            direction = offset / dist
            # clamp the magnitude expression only dist stays not modified so `direction` remains a true unit vector for labeling
            force_magnitude = min(1.0 / dist, sep_force_cap) #max is mathematically equivalent to 1/d law as well.
            total_force += direction * force_magnitude
            neighbor_count += 1 
    if neighbor_count == 0:
            return np.array([0.0, 0.0])      
    return total_force / neighbor_count

def calculate_alignment(boid_id, positions, velocities, alignment_radius, bounds):
    my_pos = positions[boid_id]
    my_velocity = velocities[boid_id]
    total_force = np.zeros(2)
    neighbor_count = 0
    for i, neighbor_pos in enumerate(positions):
        if i == boid_id:
            continue
        offset = get_torus(my_pos,neighbor_pos,bounds)
        dist = np.linalg.norm(offset)
        if dist < alignment_radius: #deliberately left out >0 since a bird on top of another is harmless, unlike /dist in sep
            total_force += velocities[i]
            neighbor_count += 1
    if neighbor_count == 0:
        return np.array([0.0, 0.0])      
    return total_force / neighbor_count - my_velocity #otherwise mean = my_velo; force = 0 (body moving in one direction)

def calculate_cohesion(boid_id, positions, cohesion_radius, bounds):
    my_pos = positions[boid_id]
    total_force = np.zeros(2)
    neighbor_count = 0
    for i, neighbor_pos in enumerate(positions):
        if i == boid_id:
            continue
        offset = get_torus(neighbor_pos, my_pos, bounds) #arguments swapped => vector pointing me from me to my neighbor
        dist = np.linalg.norm(offset)
        if dist < cohesion_radius:
            total_force += offset
            neighbor_count += 1
    if neighbor_count == 0: 
        return np.array([0.0, 0.0])
    mean = total_force / neighbor_count
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
    N = len(positions) #this gets birds count
    sep_forces = np.zeros((N, 2))
    ali_forces = np.zeros((N, 2))
    coh_forces = np.zeros((N,2))
    for i in range(N): #one pass per bird, all rules read the same unmutated state
        sep_forces[i] = calculate_separation(i, positions, physics.separation_radius, bounds, physics.sep_force_cap)
        ali_forces[i] = calculate_alignment(i, positions, velocities, physics.alignment_radius, bounds)
        coh_forces[i] = calculate_cohesion(i, positions ,physics.cohesion_radius, bounds)
    forces = {
    "separation": sep_forces * physics.w_sep,
    "alignment": ali_forces * physics.w_ali,
    "cohesion": coh_forces * physics.w_coh,}
    new_velocities = np.zeros_like(velocities, dtype=float)
    new_positions = np.zeros_like(positions, dtype=float)
    for i in range(N):
        total_force = (
                 forces["separation"][i]
               + forces["alignment"][i]
               + forces["cohesion"][i])

        v_raw = velocities[i] + (total_force * dt) #uncapped
        # FORCES ARE RECORDED PRECAP !!!!!!!!!!!!!!!!!!!!!!!!
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
