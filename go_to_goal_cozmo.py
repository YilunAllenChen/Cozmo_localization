## If you run into an "[NSApplication _setup] unrecognized selector" problem on macOS,
## try uncommenting the following snippet

# try:
#     import matplotlib
#     matplotlib.use('TkAgg')
# except ImportError:
#     pass

from skimage import color
import cozmo
import numpy as np
from numpy.linalg import inv
import threading
import time
import sys
import asyncio
from PIL import Image

from markers import detect, annotator

from grid import CozGrid
from gui import GUIWindow
from particle import Particle, Robot
from setting import *
from particle_filter import *
from utils import *

#particle filter functionality
class ParticleFilter:

    def __init__(self, grid):
        self.particles = Particle.create_random(PARTICLE_COUNT, grid)
        self.grid = grid

    def update(self, odom, r_marker_list):

        # ---------- Motion model update ----------
        self.particles = motion_update(self.particles, odom)

        # ---------- Sensor (markers) model update ----------
        self.particles = measurement_update(self.particles, r_marker_list, self.grid)

        # ---------- Show current state ----------
        # Try to find current best estimate for display
        m_x, m_y, m_h, m_confident = compute_mean_pose(self.particles)
        return (m_x, m_y, m_h, m_confident)

# tmp cache
last_pose = cozmo.util.Pose(0,0,0,angle_z=cozmo.util.Angle(degrees=0))
flag_odom_init = False

# goal location for the robot to drive to, (x, y, theta)
goal = (6,10,0)

# map
Map_filename = "map_arena.json"
grid = CozGrid(Map_filename)
gui = GUIWindow(grid, show_camera=True)
pf = ParticleFilter(grid)

def compute_odometry(curr_pose, cvt_inch=True):
    '''
    Compute the odometry given the current pose of the robot (use robot.pose)

    Input:
        - curr_pose: a cozmo.robot.Pose representing the robot's current location
        - cvt_inch: converts the odometry into grid units
    Returns:
        - 3-tuple (dx, dy, dh) representing the odometry
    '''

    global last_pose, flag_odom_init
    last_x, last_y, last_h = last_pose.position.x, last_pose.position.y, \
        last_pose.rotation.angle_z.degrees
    curr_x, curr_y, curr_h = curr_pose.position.x, curr_pose.position.y, \
        curr_pose.rotation.angle_z.degrees
    
    dx, dy = rotate_point(curr_x-last_x, curr_y-last_y, -last_h)
    if cvt_inch:
        dx, dy = dx / grid.scale, dy / grid.scale

    return (dx, dy, diff_heading_deg(curr_h, last_h))


async def marker_processing(robot, camera_settings, show_diagnostic_image=False):
    '''
    Obtain the visible markers from the current frame from Cozmo's camera. 
    Since this is an async function, it must be called using await, for example:

        markers, camera_image = await marker_processing(robot, camera_settings, show_diagnostic_image=False)

    Input:
        - robot: cozmo.robot.Robot object
        - camera_settings: 3x3 matrix representing the camera calibration settings
        - show_diagnostic_image: if True, shows what the marker detector sees after processing
    Returns:
        - a list of detected markers, each being a 3-tuple (rx, ry, rh) 
          (as expected by the particle filter's measurement update)
        - a PIL Image of what Cozmo's camera sees with marker annotations
    '''

    global grid

    # Wait for the latest image from Cozmo
    image_event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

    # Convert the image to grayscale
    image = np.array(image_event.image)
    image = color.rgb2gray(image)
    
    # Detect the markers
    markers, diag = detect.detect_markers(image, camera_settings, include_diagnostics=True)

    # Measured marker list for the particle filter, scaled by the grid scale
    marker_list = [marker['xyh'] for marker in markers]
    marker_list = [(x/grid.scale, y/grid.scale, h) for x,y,h in marker_list]

    # Annotate the camera image with the markers
    if not show_diagnostic_image:
        annotated_image = image_event.image.resize((image.shape[1] * 2, image.shape[0] * 2))
        annotator.annotate_markers(annotated_image, markers, scale=2)
    else:
        diag_image = color.gray2rgb(diag['filtered_image'])
        diag_image = Image.fromarray(np.uint8(diag_image * 255)).resize((image.shape[1] * 2, image.shape[0] * 2))
        annotator.annotate_markers(diag_image, markers, scale=2)
        annotated_image = diag_image

    return marker_list, annotated_image


async def run(robot: cozmo.robot.Robot):

    global flag_odom_init, last_pose
    global grid, gui, pf

    # start streaming
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = False
    robot.camera.enable_auto_exposure()
    await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()

    # Obtain the camera intrinsics matrix
    fx, fy = robot.camera.config.focal_length.x_y
    cx, cy = robot.camera.config.center.x_y
    camera_settings = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float)
            
    ###################

    # YOUR CODE HERE
    reached_goal = False
    goal_inch_x, goal_inch_y, goal_h = goal  # (6, 10, 0) in inches
    goal_x = goal_inch_x * 25.4 / grid.scale
    goal_y = goal_inch_y * 25.4 / grid.scale
    while True:
        # check if cozmo is in the air right now
        if robot.is_picked_up:
            robot.stop_all_motors()
            pf = ParticleFilter(grid)
            await robot.play_anim_trigger(cozmo.anim.Triggers.CubeMovedUpset).wait_for_completed()
            continue
        converged = False
        reached_goal = False
        await robot.set_head_angle(cozmo.util.degrees(5)).wait_for_completed()
        # Obtain odometry information
        odo_update = compute_odometry(robot.pose)
        # Obtain list of currently seen markers and their poses
        marker_list, annotated_image = await marker_processing(robot, camera_settings)
        # update the particle filter using the above information
        current_best = pf.update(odo_update, marker_list)
        last_pose = robot.pose
        # update the particle filter GUI
        gui.show_particles(pf.particles)
        gui.show_mean(current_best[0], current_best[1], current_best[2], current_best[3])
        gui.show_camera_image(annotated_image)
        gui.updated.set()
        # determine the robot's current actions based on the current state of
        # the localization system. actively look around if the localization has
        # not converged; drive to goal if localization has converged
        curr_x, curr_y, curr_h, curr_confident = current_best

        if curr_confident:
            converged = True

        if not converged:
            await robot.drive_wheels(25, 8)
        else:  # if converged
            robot.stop_all_motors()
            await robot.drive_wheels(-25, -8, duration=0.5)

            if not reached_goal:
                if robot.is_picked_up:
                    robot.stop_all_motors()
                    pf = ParticleFilter(grid)
                    await robot.play_anim_trigger(cozmo.anim.Triggers.CubeMovedUpset).wait_for_completed()
                    continue

                dist_x = goal_x - curr_x
                dist_y = goal_y - curr_y
                dist_diagonal = math.sqrt(dist_x ** 2 + dist_y ** 2)
                dist_diagonal_mm = dist_diagonal * grid.scale

                heading_angle = math.degrees(math.atan2(dist_y, dist_x))
                # time.sleep(1)
                await robot.turn_in_place(cozmo.util.degrees(-curr_h),
                                          speed=cozmo.util.Angle(degrees=30)).wait_for_completed()

                # time.sleep(1)
                await robot.turn_in_place(cozmo.util.degrees(heading_angle),
                                          speed=cozmo.util.Angle(degrees=30)).wait_for_completed()
                # time.sleep(1)
                await robot.drive_straight(cozmo.util.distance_mm(dist_diagonal_mm + 3),
                                           speed_mmps(30)).wait_for_completed()
                # time.sleep(1)
                if robot.is_picked_up:
                    robot.stop_all_motors()
                    pf = ParticleFilter(grid)
                    await robot.play_anim_trigger(cozmo.anim.Triggers.CubeMovedUpset).wait_for_completed()
                if robot.is_picked_up:
                    continue

                await robot.turn_in_place(cozmo.util.degrees(-heading_angle),
                                          speed=cozmo.util.Angle(degrees=30)).wait_for_completed()
                # time.sleep(1)

                if robot.is_picked_up:
                    robot.stop_all_motors()
                    pf = ParticleFilter(grid)
                    await robot.play_anim_trigger(cozmo.anim.Triggers.CubeMovedUpset).wait_for_completed()
                if robot.is_picked_up:
                    continue

                await robot.play_anim_trigger(cozmo.anim.Triggers.CodeLabHappy).wait_for_completed()
                robot.stop_all_motors()
                reached_goal = True
                pf = ParticleFilter(grid)
            else:  # converged and reached goal
                if robot.is_picked_up:
                    robot.stop_all_motors()
                    pf = ParticleFilter(grid)
                    await robot.play_anim_trigger(cozmo.anim.Triggers.CubeMovedUpset).wait_for_completed()
                    continue
                else:
                    await robot.play_anim_trigger(cozmo.anim.Triggers.CodeLabHappy).wait_for_completed()
                    robot.stop_all_motors()
    ###################

class CozmoThread(threading.Thread):
    
    def __init__(self):
        threading.Thread.__init__(self, daemon=False)

    def run(self):
        cozmo.robot.Robot.drive_off_charger_on_connect = False  # Cozmo can stay on his charger
        cozmo.run_program(run, use_viewer=False)


if __name__ == '__main__':

    # cozmo thread
    cozmo_thread = CozmoThread()
    cozmo_thread.start()

    # init
    gui.show_particles(pf.particles)
    gui.show_mean(0, 0, 0)
    gui.start()

