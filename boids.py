
import numpy as np


def get_torus(pos_a,pos_b,bounds):
    delta = pos_a - pos_b
    #confusing, it looks like a congruence to b over bound
    delta = (delta + bounds/2) % bounds - bounds/2
    return delta

        

def calculate_separation(boid_id, positions, separation_radius, bounds):
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
            force_magnitude = 1.0 / dist
            total_force += direction * force_magnitude
            neighbor_count += 1 
    if neighbor_count == 0:
            return np.array([0.0, 0.0])      
    return total_force / neighbor_count


positions = np.array([
    [10.0, 10.0],  # bird 0
    [15.0, 10.0],  # bird 1
    [80.0, 80.0],  # bird 2
])
for i in range(3):
    force = calculate_separation(i, positions, separation_radius= 20.0, bounds= 100.0)

    print(f"Bird {i}: force = {force}")