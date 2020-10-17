import anki_vector
from anki_vector.events import Events
from anki_vector import annotate
from anki_vector.util import degrees, distance_mm, distance_inches, speed_mmps
import time
import numpy as np
from skimage import color
from PIL import ImageDraw, ImageFont
from markers import detect, annotator_vector

stop = False

def main():
    args = anki_vector.util.parse_command_args()

    camera_settings = np.array([
        [296.54,      0, 160],    # fx   0  cx
        [     0, 296.54, 120],    #  0  fy  cy
        [     0,      0,   1]     #  0   0   1
    ], dtype=np.float)

    with anki_vector.Robot(serial=args.serial,show_viewer=True) as robot:
        robot.camera.init_camera_feed()
        robot.behavior.set_head_angle(degrees(0))
        marker_annotate = annotator_vector.MarkerAnnotator(robot.camera.image_annotator)
        robot.camera.image_annotator.add_annotator('Marker', marker_annotate)
        while True:
            time.sleep(0.05)
            if not robot.camera.latest_image:
                continue
            # Get the latest image from Vector and convert it to grayscale
            new_image = robot.camera.latest_image.raw_image
            image = np.array(new_image)
            image = color.rgb2gray(image)
            
            # Detect the marker
            markers = detect.detect_markers(image, camera_settings)

            marker_annotate.markers = markers
                     
            # Process each marker
            for marker in markers:
                
                # Get the cropped, unwarped image of just the marker
                marker_image = marker['unwarped_image']

                # ...
                # label = my_classifier_function(marker_iamge)
                
                # Get the estimated location/heading of the marker
                pose = marker['pose']
                x, y, h = marker['xyh'] 

                print('X: {:0.2f} mm'.format(x))
                print('Y: {:0.2f} mm'.format(y))
                print('H: {:0.2f} deg'.format(h))
                print()

# try:
#     stop = False
#     cozmo.run_program(run, use_viewer=True)
# except cozmo.exceptions.ConnectionError:
#     print('Could not connect to Cozmo')
# except KeyboardInterrupt:
#     print('Stopped by user')
#     stop = True
if __name__ == "__main__":
    main()