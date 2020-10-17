from grid import CozGrid
from particle import Particle
from utils import grid_distance, rotate_point, diff_heading_deg, add_odometry_noise
import setting
import math
import numpy as np

from pprint import pformat as pf
from utils import compute_mean_pose as cpmean
from time import sleep


def neighbor(ptc1: tuple, ptc2: tuple) -> bool:
    '''
    Function determines whether the two points are close enough.

    :param ptc1: The first point in form (x, y, heading)
    :param ptc2: The second point in form (x, y, heading)
    '''
    threshold = 1.5
    if abs(ptc1[0] - ptc2[0]) >= threshold or abs(ptc1[1] - ptc2[1]) >= threshold:
        return False
    if abs(ptc1[2] - ptc2[2]) >= 15:
        return False
    return True


def approx(particle: object, ptc2_observation: list, grid: object) -> bool:
    '''
    Function is inspired by merge sort (to cope with wrong readings) and uses a score-based approach to determine whether a particle aligns with the robot's observation

    :param particle: The particle
    :param ptc2_observation: The observations of the robot
    :param grid: Global grid
    '''
    ptc1_observation = sorted(
        particle.read_markers(grid), key=lambda obs: obs[0])
    ptc2_observation = sorted(ptc2_observation, key=lambda obs: obs[0])
    if len(ptc1_observation) == 0 and len(ptc2_observation) == 0:
        return True
    ptr1, ptr2 = 0, 0
    score = 0
    while(ptr1 < len(ptc1_observation) and ptr2 < len(ptc2_observation)):
        ptc1, ptc2 = ptc1_observation[ptr1], ptc2_observation[ptr2]
        if neighbor(ptc1, ptc2):
            ptr1 += 1
            ptr2 += 1
            score += 3
        else:
            if ptc1[0] > ptc2[0]:
                ptr2 += 1
            else:
                ptr1 += 1
    score -= (len(ptc1_observation) + len(ptc2_observation))
    return score > 0



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


def measurement_update(particles, measured_marker_list, grid):
    """ Particle filter measurement update

        Arguments:
        particles -- input list of particle represents belief \tilde{p}(x_{t} | u_{t})
                before measurement update (but after motion update)

        measured_marker_list -- robot detected marker list, each marker has format:
                measured_marker_list[i] = (rx, ry, rh)
                rx -- marker's relative X coordinate in robot's frame
                ry -- marker's relative Y coordinate in robot's frame
                rh -- marker's relative heading in robot's frame, in degree

                * Note that the robot can only see markers which is in its camera field of view,
                which is defined by ROBOT_CAMERA_FOV_DEG in setting.py
                                * Note that the robot can see mutliple markers at once, and may not see any one

        grid -- grid world map, which contains the marker information,
                see grid.py and CozGrid for definition
                Can be used to evaluate particles

        Returns: the list of particles represents belief p(x_{t} | u_{t})
                after measurement update
    """

    # Remove invalid points
    for p in particles:
        x, y = p.xy
        if not grid.is_in(x, y) or not grid.is_free(x, y):
            particles.remove(p)

    # Re-populate the points as needed
    re_sampling = Particle.create_random(5000-len(particles),grid)
    repop_particles = particles + re_sampling
    filtered = list(filter(lambda particle: approx(
        particle, measured_marker_list, grid), particles))
    
    if len(filtered) == 0:
        re_sampling = Particle.create_random(5000-len(particles),grid)
        return particles + re_sampling
    elif(len(filtered) < 2500):
        while(len(filtered) < 2500):
        # Increase weights of existing good points
            filtered = filtered * 2
    return filtered
