

import cv2
import edge
import numpy as np
import matplotlib.pyplot as plt
import filter
import data

class Lane():
    fig_hist = plt.figure(1)
    plt.ion()

    def __init__(self, px_width, px_heidht, focal_point=None, roi_height=None, source_pts=None, lane_width=3.7, lane_length=24, queue_size=32):
        # initialises common variables in the class
        # focal_point : location of the focal point of the lane. Can be the
        #               vanishing point of the image
        # roi_height : height where the lane region of interest is at most
        #              considered
        # source_pts : bottom start points of the lane roi
        # lane_width : physical measurement spacing between road lines. Default = 3.7m

        if focal_point is None:
            self.focal_point = [0,0]
        else:
            self.focal_point = focal_point

        if roi_height is None:
            self.roi_height = 0.
        else:
            self.roi_height = roi_height

        if source_pts is None:
            self.source_pts = [[0, 0], [0, 0]]
        else:
            self.source_pts = source_pts

        self.roi_pts = np.float32([[0, 0], [0, 0], [0, 0], [0, 0]])
        self.left_fit = None
        self.right_fit = None

        self.h = px_heidht  # vertical pixel count of the camera
        self.w = px_width  # horizontal pixel count of the camera

        self.lane_width = lane_width # dimension of lane width in meters
        self.width_per_pix = 0
        self.lane_length = lane_length # dimension of lane length in meters
        self.len_per_pix = 0

        self.left_pts = None
        self.right_pts = None
        self.center_pts = None
        self.y_pts = None

        self.queue_size = queue_size
        self.left_fit_filter = filter.Filter(self.queue_size)
        self.right_fit_filter = filter.Filter(self.queue_size)
        self.rad_filter = filter.Filter(self.queue_size)
        self.car_center_pos = filter.Filter(self.queue_size)
        self.corner_rad = 0

    def lane_roi(self, roi_height=None, focal_point=None, source_pts=None):
        # defines a lanes region of interest
        # img_shape : shape of the input image
        # roi_height : value between (0 -> 1) the pixel height of the highest
        # point of interest with respect from the bottom of the image.
        # focal_point : location of the focal focal_point. If None, will use
        #               the predefined focal_point.
        # source_pts : location of the two bottom corner points
        # return : coordinates of the region of interest of a lane

        if focal_point is None:
            focal_point = self.focal_point

        if roi_height is None:
            roi_height = self.roi_height

        # top of the roi is a factor of the height from the bottom of the roi
        # to the focal point.
        # (img_height - focal_height)*(1 - desired_ratio_height) + focal_height
        # h_top is the y position of the height with respect to the difference
        # between the image height and the focal point.
        fph = self.focal_point[1]  # height of focal point
        h_top = (self.h - fph)*(1 - roi_height) + fph

        if source_pts is None:
            # create the source points as the two bottom corners of the image
            source_pts = self.source_pts

        m_left = (focal_point[1] - source_pts[0][1]) / (focal_point[0] - source_pts[0][0])
        b_left = focal_point[1] - (m_left * focal_point[0])
        x_left = (h_top - b_left) // m_left

        m_right = (focal_point[1] - source_pts[1][1]) / (focal_point[0] - source_pts[1][0])
        b_right = focal_point[1] - (m_right * focal_point[0])
        x_right = (h_top - b_right) // m_right

        self.roi_pts = np.float32([source_pts[0], [x_left, h_top], [x_right, h_top], source_pts[1]])
        return self.roi_pts

    def draw_lane_roi(self, img, roi_pts=None, focal_point=None, color=(255, 255, 255)):
        # draws the region of interest onto the supplied image
        # img : source image
        # roi_pts : coordinate points of the region of interest
        # focal_point : location of the focal focal_point
        # return : the supplied image with the roi drawn on

        if focal_point is None:
            focal_point = self.focal_point
        if roi_pts is None:
            roi_pts = self.roi_pts

        image = img.copy()
        pts = np.int32(roi_pts)
        pts = pts.reshape((-1, 1, 2))
        cv2.circle(image, (focal_point[0], focal_point[1]), 5, color, 2)
        cv2.polylines(image, [pts], True, color, 2)

        return image

    def warp_image(self, img, roi_pts=None, location_pts=None, padding=(0,0)):
        # img : image to be transformed into the new perspective
        # roi_pts : location points from the original image to be transformed.
        #           Points must be in a clock wise order.
        # location_pts : the final location points in the image where the
        #           old_pts will be located. If None supplied, the new points
        #           will be the four corners off the supplied image in a
        #           clockwise order, starting at point (0,0).
        # offset : adds padding onto the roi points so the warped image is
        #          larger than the roi. Supplied as (width, height) padding
        # returns : the warped perspective image with the supplied points

        if roi_pts is None:
            roi_pts = self.roi_pts

        if location_pts is None:
            location_pts = np.float32([[padding[0], self.h-padding[1]], # bot-left
                                       [padding[0], padding[1]], # top-left
                                       [self.w-padding[0], padding[1]], # top-right
                                       [self.w-padding[0], self.h-padding[1]]]) # bot-right

        # calculate the perspective transform matrix between the old and new points
        self.M = cv2.getPerspectiveTransform(roi_pts, location_pts)
        # Warp the image to the new perspective
        return cv2.warpPerspective(img, self.M, (self.w, self.h))

    def inverse_warp_image(self, img):
        # Unwarp a warped image to the old perspective
        # img : warped image to be unwarped
        # return : warped image in old perspective
        return cv2.warpPerspective(img, self.M, (self.w, self.h), flags=cv2.WARP_INVERSE_MAP)

    def mask_roi(self, img, roi_pts=None, outside_mask=True):
        # create a masked image showing only the area of the roi_pts
        # img : source image to be masked
        # roi_pts : region for masking
        # outside_mask : True if masking area outside roi, False if masking roi
        # return : masked image

        if roi_pts is None:
            roi_pts = self.roi_pts

        pts = np.int32(roi_pts)
        pts = [pts.reshape((-1, 1, 2))]
        # return the applyed mask
        if outside_mask == True:
            mask = np.zeros_like(img)
            ignore_mask_color = (1, 1, 1)
        else:
            mask = np.ones_like(img)
            ignore_mask_color = (0, 0, 0)

        # create a polygon that is white
        poly_mask = cv2.fillPoly(mask, pts, ignore_mask_color)
        return img * poly_mask

    def binary_image(self, arg, *argv):
        # combines multiple binary vectors together to create one binary vector
        # arg : first binary vector
        # *argv : other binary vectors to be be combined
        # return : binary vector same shape as arg

        # Combine the two binary thresholds
        combined = arg
        for arg_vect in argv:
            combined[(combined == 1) | (arg_vect == 1)] = 1

        return combined

    def combine_images(self, img_one, img_two, img_one_weight=0.8, img_two_weight=1.):
        # combines two images into one for display purposes
        # img_one : image one
        # img_two : image two
        # img_one_weight : transparency weight of image one
        # img_two_weight : transparency weight of image two
        # return : combined image
        return cv2.addWeighted(img_one, img_one_weight, img_two, img_two_weight, 0)

    def gauss(self, x, mu, sigma, A):
        # creates a gaussian distribution from the data
        # x : input data
        # mu : mean data point
        # sigma : variance from the mean
        # return : Gaussian distribution
        return A * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2)

    def bimodal(self, x, mu1, sigma1, A1, mu2, sigma2, A2):
        return self.gauss(x, mu1, sigma1, A1) + self.gauss(x, mu2, sigma2, A2)

    def plot_graph(self, x_data, y_data):
        # plot a real time histogram of the supplied data
        # x_data : X data to plot
        # y_data : Y data to plot
        plt.clf()
        for y in y_data:
            plt.plot(y, x_data)
        plt.gca().invert_yaxis()  # to visualize as we do the images
        plt.pause(0.00001)

    def plot_histogram(self, data):
        # plot a real time histogram of the supplied data
        # data : data to plot
        plt.clf()
        plt.plot(data)
        plt.pause(0.00001)

    def quadratic_line(self, start, stop, *argv, calc='x', plot=False):
        # gets each pixle value for a quadratic over the image
        # start : start value for the quadratic
        # stop : end value for the quadratic
        # calc : what quadratic variable do calculate
        # quad_values : list of quadratic variables
        # result : list of x and y coordinates

        data = []
        count = 0
        array = np.array([n for n in range(start, stop)])
        for arg_vect in argv:
            if len(arg_vect) != 3:
                raise 'there must be 3 quadratic values supplied'

            data.append(arg_vect[0] * array ** 2 + arg_vect[1] * array + arg_vect[2])
            count += 1

        # make the last list the average of all the lists
        data.append(np.sum(data, axis=0) / count)

        lines = np.array(data)
        self.left_pts = lines[0]
        self.right_pts = lines[1]
        self.center_pts = lines[2]
        self.y_pts = array

        if plot:
            self.plot_graph(array, lines)
        return lines, array

    def lane_lines_radius(self, y_max):
        # get the radius of the lane function with respect to the center of the lane
        # y_max : the distance of the function to be calculated
        # return : the radius of the road in m

        # get the average of the two function parameters
        cent_fit = (self.left_fit + self.right_fit) / 2

        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        rad = ((1 + (2 * cent_fit[0] * y_max * self.len_per_pix + cent_fit[1]) ** 2) ** 1.5) / np.absolute(2 * cent_fit[0])
        return rad/2

    def histogram(self, data):
        # calculates the histogram of data
        # data : data to be transformed into a histogram
        # returns : a vector of the histogram data
        return np.sum(data, axis=0)

    def histogram_peaks(self, data, plot_hist=False):
        # finds the peak location of a data line
        # data : input 2D data to locate peaks
        # plot_hist : plot the histogram of the data
        # return : the peak locations
        hist = self.histogram(data)

        if plot_hist == True:
            self.plot_histogram(hist)

        midpoint = np.int(hist.shape[0] // 2)
        leftx_base = np.argmax(hist[:midpoint])
        rightx_base = np.argmax(hist[midpoint:]) + midpoint

        return leftx_base, rightx_base

    def plot_best_fit(self, img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, margin=100):
        # plots the search boxes and lane lines of where the lane lines are thought to be

        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((img, img, img)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = self.combine_images(out_img, window_img, img_one_weight=1, img_two_weight=0.3)

        cv2.imshow('result', result) # visulise the output of the function

    def find_lane_lines(self, img, line_windows=10, plot_line=False, draw_square=False):
        # finds the lane line locations within an image
        # img : image that needs o be search for line. Should be a warped image
        # line_windows : how many windods are used in the search for the lane lines
        # plot_line : True => plots the found lines onto the image for visulisation
        # draw_square : draws the search boxes locations wher the lane lines are located
        # return : quadratics functions of the left and right lane lines

        out_img = img.copy()

        # Set height of windows
        window_height = np.int(img.shape[0] / line_windows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx, rightx = self.histogram_peaks(img)
        leftx_current = leftx
        rightx_current = rightx
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(line_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            if draw_square == True:
                # Draw the windows on the visualization image
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (255, 255, 255), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (255, 255, 255), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        try:
            self.left_fit = np.polyfit(lefty, leftx, 2)
        except:
            self.left_fit = [0,0,0]
        try:
            self.right_fit = np.polyfit(righty, rightx, 2)
        except:
            self.right_fit = [0,0,0]

        if plot_line==True:
            # plot the line of best fit onto the image
            self.plot_best_fit(out_img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds)

        return self.left_fit, self.right_fit

    def refresh_lane_lines(self, img, plot_line=False):
        # uses the previous lane line locations to refresh the current lane line location
        # img : image that needs o be search for line. Should be a warped image
        # plot_line : True => plots the found lines onto the image for visulisation
        # return : quadratics functions of the left and right lane lines

        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = (
            (nonzerox > (
            self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy + self.left_fit[2] - margin)) & (
                nonzerox < (
                self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy + self.left_fit[2] + margin)))
        right_lane_inds = (
            (nonzerox > (
            self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy + self.right_fit[2] - margin)) & (
                nonzerox < (
                self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy + self.right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

        if plot_line == True:
            # plot the line of best fit onto the image
            self.plot_best_fit(img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds)

        return self.left_fit, self.right_fit

    def lane_lines(self, image, plot_line=False):
        # finds the location of the lanes lines
        # image : image that needs o be search for line. Should be a warped image
        # plot_line : True => plots the found lines onto the image for visulisation
        # return : quadratics functions of the left and right lane lines

        # is the input image a binary image or a multi-channel image
        if len(image.shape) > 2:
            # image has multiple channels. convert the image to a binary image
            # raise 'Lane.lane_lines input image needs to be a binary image'
            img = image[:,:,0]
            for channel in range(1, image.shape[2]):
                img = self.binary_image(img, image[:,:,channel])
        else:
            img = image.copy()

        # Does the program know where the lane lines are?
        if self.left_fit is None or self.right_fit is None:
            # Don't know where the lane lines are, so go and find them
            self.find_lane_lines(img, plot_line=plot_line, draw_square=plot_line)
        else:
            self.refresh_lane_lines(img, plot_line=plot_line)

        return self.left_fit, self.right_fit

    def driving_lane(self, image):
        # function that finds the road driving lane line
        # image : camera image where the line locations are to be located
        # return : a masked image of onlt the lane lines

        # Convert to HSV color space and separate the V channel
        # hls for Sobel edge detection
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        # use on the luminance channel data for edges
        # TODO: implement edge.binary_array in pycuda for speed
        _, sxbinary = edge.threshold(hls[:, :, 1], thresh=(120, 255))
        sxbinary = edge.blur_gaussian(sxbinary, ksize=3)

        # find the edges in the channel data using sobel magnitudes
        sxbinary = edge.mag_thresh(sxbinary, sobel_kernel=3, thresh=(110, 255))

        s_channel = hls[:, :, 2]  # use only the saturation channel data
        _, s_binary = edge.threshold(s_channel, (110, 255))
        _, r_thresh = edge.threshold(image[:, :, 2], thresh=(120, 255))

        rs_binary = cv2.bitwise_and(s_binary, r_thresh)
        return cv2.bitwise_or(rs_binary, sxbinary.astype(np.uint8))

    def car_lane_pos(self):
        # find the relative position of the car with respect to the center of
        # the driving lane
        # return : distance left of center in meters

        if self.width_per_pix == 0:
            # find the pixel count per meter in the horizontal direction
            x_left_start = self.left_pts[-1]  # start point of the left lane line
            x_right_start = self.right_pts[-1]
            total_pix = np.absolute(x_right_start - x_left_start)
            self.width_per_pix = self.lane_width / total_pix

        if self.len_per_pix == 0:
            # find the pixel count per meter in the horizontal direction
            self.len_per_pix = self.lane_length / self.h

        # +ve driving more on the left side of the lane
        car_off_centre = self.center_pts[-1] - self.w // 2
        return car_off_centre * self.width_per_pix

    def overlay_lane(self, img, color=(0,255,0), overlay_weight=0.3):
        # combines the found road lane with the camera image
        # img : original camera image
        # color : overlay color
        # overlay_weight : weight factor of transparency of the overlay
        # return : image with the over-layed lane

        # Create an image to draw the lines on
        color_warp = np.zeros_like(img).astype(np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_pts, self.y_pts]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_pts, self.y_pts])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), color)

        # Warp the blank back to original image space using inverse perspective matrix
        newwarp = self.inverse_warp_image(color_warp)
        # Combine the result with the original image
        return cv2.addWeighted(img, 1, newwarp, overlay_weight, 0)

    def display_text(self, img, text, pos, color=(255, 255, 255)):
        # adds text to the image
        # img : image that will have text added
        # text : text to add
        # pos : position of the text
        # color : colour of the text
        # return : image with text overlayed

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, pos, font, 1, color, 2, cv2.LINE_AA)

        return img
