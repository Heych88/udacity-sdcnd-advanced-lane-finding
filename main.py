"""
this program takes in a checkerboard image from a camera and calibrates the
image to remove camera radial and tangential distortion.
"""

import cv2
from driveline import Lane
from camera import CameraImage, CameraCalibration
import data

ci = CameraImage()

# get the undistorted camera matrix
cal = CameraCalibration()
check_img = cv2.imread('camera_cal/calibration2.jpg', -1)
cal.get_optimal_calibration(check_img, chess_count=(9,6))
# undistort the calibration image for verification
undistort_img = cal.undistort_image(check_img)
cv2.imwrite('output_images/undistorted_calibration2.jpg', undistort_img)

# Define the codec and create VideoWriter object
if data.isVideo:
    # setup video recording when using a video
    fourcc = cv2.VideoWriter_fourcc(* 'MJPG')
    filename = 'output_images/' + data.video
    out = cv2.VideoWriter(filename, fourcc, 20.0, (1280, 720))

if data.isVideo:
    cam = cv2.VideoCapture(data.img_add)
    
def lane_pipeline(img, show_roi=False, show_lane=True, display_curve=True):
    # pipeline for the current driving lane
    # img : image from the camera in which we need to locate the lane lines
    # show_roi : True => will show the warped image location area of interest
    # show_lane : True => shous the overlay of the lane
    # display_curve : shows the road radius and vehicle position relative to
    #                 the lane, onto the image
    # return : original image with added features and fount lane location

    # find the edges in the image for line detection
    pipe_img = drive_lane.driving_lane(img)
    # Warp the image using OpenCV warpPerspective()
    warped = drive_lane.warp_image(pipe_img)

    # get the polynomials of the lane lines
    left_fit, right_fit = drive_lane.lane_lines(warped, plot_line=False)

    # get the moving average for the line data
    drive_lane.left_fit = drive_lane.left_fit_filter.moving_average(left_fit)
    drive_lane.right_fit = drive_lane.right_fit_filter.moving_average(right_fit)

    # find the x and y points for the found quadratics
    drive_lane.quadratic_line(0, drive_lane.h, left_fit, right_fit, plot=False)

    if show_roi:
        # draw the region of interest on the image
        img = drive_lane.draw_lane_roi(img, color=(255, 0, 0))

    if show_lane:
        # draw the overlay onto the image
        img = drive_lane.overlay_lane(img)

    if display_curve:
        try:
            c_rad = drive_lane.lane_lines_radius(drive_lane.h)
            drive_lane.corner_rad = drive_lane.rad_filter.moving_average(c_rad)
        except:
            drive_lane.corner_rad = 0

        car_off_center = drive_lane.car_lane_pos()

        # add the curviture and vehicle position to the image
        drive_lane.display_text(img, 'Curve Radius : {:.0f}m'.format(drive_lane.corner_rad), (10, 40))
        drive_lane.display_text(img, 'Left of center : {:.3f}m'.format(car_off_center), (10, 90))
        #stdout.write("\r{:.3f}m Left of Center, Corner Radius : {:.0f}m".format(car_off_center, self.corner_rad))

    return img

while(1):
    # continually loop if the input is a video until it ends of the user presses 'q'
    # if an image execute once and wait till the user presses a key
    if data.isVideo:
        ret, image = cam.read()
        if ret == False:
            break
    else:
        # read in the image to the program
        image = cv2.imread(data.img_add, -1)
        image = cal.undistort_image(image)
        cv2.imwrite('output_images/undistorted_car.jpg', image)

    # see if we have already defined the class drive_lane
    try:
        drive_lane
    except NameError:
        # first frame call. lets define the variable center_lane
        h, w = image.shape[:2]

        focal_point = [w // 2, int(h * data.fph)]
        drive_lane = Lane(w, h, focal_point=focal_point, roi_height=data.roi_height,
                           source_pts=[[0, h], [w, h]], lane_length=data.phm)
        # setup the region of interest of the image denoting were the lane
        # will most likely be located from the above coordinates
        src = drive_lane.lane_roi()

    # all the lane calculations for the current driving lane are done here
    lane_img = lane_pipeline(image)
    cv2.imshow('final', lane_img)

    # wait for a user key interrupt then close all windows
    if data.isVideo:
        out.write(lane_img)  # save image to video

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # save the new image
        cv2.imwrite('output_images/roi_' + data.image, lane_img)
        cv2.waitKey(0)
        break

if data.isVideo:
    out.release()
    cam.release()

cv2.destroyAllWindows()
