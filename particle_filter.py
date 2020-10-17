from itertools import product
from grid import *
from particle import Particle
from utils import *
import setting
import numpy as np
np.random.seed(setting.RANDOM_SEED)


def motion_update(particles, odom):
    """ Particle filter motion update
        Arguments:
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- odometry to move (dx, dy, dh) in *robot local frame*
        Returns: the list of particles represents belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """

    motion_particles = []
    dx, dy, dh = odom
    for p in particles:
        x, y, h = p.xyh
        new_dx, new_dy = rotate_point(dx, dy, h)
        x += new_dx
        y += new_dy
        h += dh
        curr_x, curr_y, curr_h = add_odometry_noise((x, y, h), setting.ODOM_HEAD_SIGMA, setting.ODOM_TRANS_SIGMA)
        motion_particles.append(Particle(curr_x, curr_y, curr_h))
    return motion_particles

# ------------------------------------------------------------------------

# # Legacy solution - doesn't converge. :(
# def measurement_update(particles, measured_marker_list, grid):
#     """ Particle filter measurement update

#         Arguments:
#         particles -- input list of particle represents belief \tilde{p}(x_{t} | u_{t})
#                 before measurement update (but after motion update)

#         measured_marker_list -- robot detected marker list, each marker has format:
#                 measured_marker_list[i] = (rx, ry, rh)
#                 rx -- marker's relative X coordinate in robot's frame
#                 ry -- marker's relative Y coordinate in robot's frame
#                 rh -- marker's relative heading in robot's frame, in degree

#                 * Note that the robot can only see markers which is in its camera field of view,
#                 which is defined by ROBOT_CAMERA_FOV_DEG in setting.py
#                                 * Note that the robot can see mutliple markers at once, and may not see any one

#         grid -- grid world map, which contains the marker information,
#                 see grid.py and CozGrid for definition
#                 Can be used to evaluate particles

#         Returns: the list of particles represents belief p(x_{t} | u_{t})
#                 after measurement update
#     """

#    # Remove invalid points
#     for p in particles:
#         x, y = p.xy
#         if not grid.is_in(x, y) or not grid.is_free(x, y):
#             particles.remove(p)
#     print(len(particles))
#     # Re-populate the points as needed
#     re_sampling = Particle.create_random(5000-len(particles),grid)
#     repop_particles = particles + re_sampling
#     filtered = list(filter(lambda particle: approx(
#         particle, measured_marker_list, grid), particles))
    
#     if len(filtered) == 0:
#         re_sampling = Particle.create_random(5000-len(particles),grid)
#         return particles + re_sampling
#     else:
#         exploration_ratio = 0.01
#         for i in range(int(len(particles) * exploration_ratio * 20)):
#             particles.remove(choice(particles))
#         for i in range(int((5000 - len(particles)) * (1-exploration_ratio))):
#             particles.append(choice(filtered))
#         random_exploration = Particle.create_random(int((5000 - len(particles)) * exploration_ratio), grid)
#         return particles + random_exploration
#     return particles





def measurement_update(particles, measured_marker_list, grid):
    """ Particle filter measurement update
        Arguments:
        particles -- input list of particle represents belief \tilde{p}(x_{t} | u_{t})
                before meansurement update (but after motion update)
        measured_marker_list -- robot detected marker list, each marker has for_markerat:
                measured_marker_list[i] = (rx, ry, rh)
                rx -- marker's relative X coordinate in robot's frame
                ry -- marker's relative Y coordinate in robot's frame
                rh -- marker's relative heading in robot's frame, in degree
                * Note that the robot can only see markers which is in its camera field of view,
                which is defined by ROBOT_CAMERA_FOV_DEG in setting.py
                                * Note that the robot can see mutliple markers at once, and may not see any one
        grid -- grid world map, which contains the marker infor_markeration,
                see grid.py and CozGrid for definition
                Can be used to evaluate particles
        Returns: the list of particles represents belief p(x_{t} | u_{t})
                after measurement update
    """
    EXPLORATION_NUM = 25
    measured_particles = []
    weights = []

    if len(measured_marker_list) > 0:
        for particle in particles:
            x, y = particle.xy
            if grid.is_in(x, y) and grid.is_free(x, y):
                probs = []
                p_markers = particle.read_markers(grid)
                r_markers = measured_marker_list.copy()
                while len(r_markers) > 0 and len(p_markers) > 0 :
                    # Iterate through all particle markers/robot markers pairs and select the best particles
                    pr_pairs = product(p_markers, r_markers)
                    p_marker, r_marker = min(pr_pairs, key=lambda pos: grid_distance(
                        pos[0][0], pos[0][1], pos[1][0], pos[1][1]))

                    # Compute their relative weights using their deviation from ground truth values (position, heading)
                    # The closer the particles are to the ground truth values, the heavier they are.
                    distance = grid_distance(
                        p_marker[0], p_marker[1], r_marker[0], r_marker[1])
                    heading_diff = diff_heading_deg(p_marker[2], r_marker[2])
                    sigma_1 = (distance**2) / (2*setting.MARKER_TRANS_SIGMA**2)
                    sigma_2 = (heading_diff**2) / (2*setting.MARKER_ROT_SIGMA**2)

                    # Store the relative weight into a temporary holder
                    probs.append(math.exp(-sigma_1-sigma_2))
                    p_markers.remove(p_marker)
                    r_markers.remove(r_marker)

                # Take into consideration false positives and true negatives
                assigned_weight = 1.
                false_prob = setting.DETECTION_FAILURE_RATE * setting.SPURIOUS_DETECTION_RATE
                for prob in probs:
                    assigned_weight = prob if prob > false_prob else false_prob
                assigned_weight *= (setting.DETECTION_FAILURE_RATE**len(p_markers)) * \
                    (setting.SPURIOUS_DETECTION_RATE**len(r_markers))
                # Assign this weight to this particle
                weights.append(assigned_weight)
            else:
                weights.append(0.0)
    else:
        weights = [1.0] * len(particles)

    # Normalize all weights so we can use it in random.choice
    normalization_factor = float(sum(weights))
    if normalization_factor != 0:
        weights = [weight/normalization_factor for weight in weights]
        measured_particles = list(np.random.choice(
            particles, setting.PARTICLE_COUNT-EXPLORATION_NUM, p=weights))
        measured_particles += Particle.create_random(EXPLORATION_NUM, grid)
    else:
        measured_particles = Particle.create_random(
            setting.PARTICLE_COUNT, grid)

    return measured_particles
