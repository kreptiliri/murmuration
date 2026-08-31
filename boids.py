
import numpy as np


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

def velocity_capper(old_velocity, max_speed):
    speed = np.linalg.norm(old_velocity)
    if speed > max_speed:
        return (old_velocity/speed) * max_speed
    return old_velocity
def step(positions, velocities, dt, separation_radius, bounds, max_speed, sep_force_cap=1.0,w_sep=1.5, w_ali=1.0, w_coh=0.3):
    N = len(positions) #this gets birds count
    sep_forces = np.zeros((N, 2))
    for i in range(N): #gets each birds sep_force
        sep_forces[i] = calculate_separation(i, positions, separation_radius, bounds, sep_force_cap)
    forces = {
    "separation": sep_forces * w_sep,
    "alignment": np.zeros((N, 2)), #future use
    "cohesion": np.zeros((N, 2))}
    new_velocities = np.zeros_like(velocities, dtype=float)
    new_positions = np.zeros_like(positions, dtype=float)
    for i in range(N):
        total_force = (forces["separation"][i]
               + forces["alignment"][i]
               + forces["cohesion"][i])

        v_raw = velocities[i] + (total_force * dt) #uncapped
        # FORCES ARE RECORDED PRECAP !!!!!!!!!!!!!!!!!!!!!!!!
        new_velocities[i] = velocity_capper(v_raw, max_speed)
        new_positions[i] = positions[i] + new_velocities[i] * dt

    new_positions %= bounds
    return new_positions, new_velocities, forces


if __name__ == "__main__":
    # test data
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

    #constants
    dt = 0.1 
    separation_radius = 20.0
    bounds = 100.0
    max_speed = 5.0 #a design thought
    next_pos, next_vel, recorded_forces = step(test_positions, test_velocities, dt, separation_radius, bounds, max_speed)

