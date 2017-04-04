import numpy as np
import cv2

def optimal_threshold(img, sigma=0.3):
    # gets optimal thresholds for the supplied image
    # img : supplied image
    # sigma : factor variance of each threshold from the computed median
    # return : lower and upper threshold values
    # compute the median of the single channel pixel intensities
    v = np.median(img)
    # get the best threshold from the computed median
    low_threshold = int(max(0, (1 - sigma) * v))
    high_threshold = int(min(255, (1 + sigma) * v))

    return low_threshold, high_threshold

def sobel(img_channel, orient='x', sobel_kernel=3):

    if orient == 'x':
        sobel = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, sobel_kernel)
    if orient == 'y':
        sobel = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, sobel_kernel)

    return sobel

def binary_array(array, thresh, value=1):
    # turns an array into a binary array when between a threshold
    # array : numpy array to be converted to binary
    # thresh : threshold values between which a change in binary is stored.
    #          Threshold is inclusive
    # value : output value when between the supplied threshold
    # return : Binary array version of the supplied array

    # Is activation (1) between the threshold values (band-pass) or is it
    # outside the threshold values (band-stop)
    if value == 0:
        # Create a binary array the same size of as the input array
        # band-stop binary array
        binary = np.ones_like(array)
    else:
        # band-pass binary array
        binary = np.zeros_like(array)
        value = 1

    binary[(array >= thresh[0]) & (array <= thresh[1])] = value
    return binary

def rescale_to_8bit(array, max_value=255):
    # rescales input array to an uint8 array
    # array : array to be rescaled
    # max_value : maximum value in the returned array
    # return : 8-bit array
    return np.uint8(max_value * array / np.max(array))

def abs_sobel_thresh(img_channel, orient='x', sobel_kernel=3, thresh=(0, 255)):

    # get the sobel of the image in the orientation supplied
    abs_sobel = np.absolute(sobel(img_channel, orient=orient, sobel_kernel=sobel_kernel))
    # return a band-pass binary array
    return binary_array(abs_sobel, thresh)

def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):

    # get the edges in the horizontal direction
    sobelx = np.absolute(sobel(image, orient='x', sobel_kernel=sobel_kernel))
    # get the edges in the vertical direction
    sobely = np.absolute(sobel(image, orient='y', sobel_kernel=sobel_kernel))

    # Calculate the edge magnitudes
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    return binary_array(mag, thresh)

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2)):

    # get the edges in the horizontal direction
    abs_sobelx = np.absolute(sobel(image, orient='x', sobel_kernel=sobel_kernel))
    # get the edges in the vertical direction
    abs_sobely = np.absolute(sobel(image, orient='y', sobel_kernel=sobel_kernel))

    gradient = np.arctan2(abs_sobely, abs_sobelx)
    return binary_array(gradient, thresh)

def channel_range_isolate(channel, val_min, val_max):
    # takes in a single channel and isolates only the values in the supplied range, then scales them to 8-bit.
    # channel : input 2D array to be isolated
    # val_min : min threshold value
    # max_val : max threshold value
    # return : scaled isolated channel

    for i in range(0, channel.shape[0]):
        for j in range(0, channel.shape[1]):
            if ((channel[i, j] < val_min) | (channel[i, j]  > val_max)):
                channel[i, j] = val_min

    return np.uint8(((channel - val_min)/(val_max - val_min)*255))

def group_channel(channel, gcount):
    # splits the channel data into groups where the data value is changed to
    # the group middle value. group is per array value between 0 -> 255.
    # example, gcount = 5, cannel value = 180,
    #          return = (int(180/(255/5))+ 0.5)*(255/5) = 178
    # channel : 8-bit data to be be grouped
    # gcount : number of groups
    # return : 8-bit array of the channel data that has been grouped
    gsize = 255/gcount
    return ((channel//gsize + 0.5)*gsize).astype(np.uint8)

def group_image(image, gcount):
    # splits each channel in an image into groups where the data value is
    # changed to the group middle value. group is per array value between
    # 0 -> 255.
    # example, gcount = 5, cannel value = 180,
    #          return = (int(180/(255/5))+ 0.5)*(255/5) = 178
    # image : multi channel 8-bit data to be be grouped
    # gcount : number of groups
    # return : 8-bit array of the channel data that has been grouped

    for channel in range(image.shape[2]):
        image[:,:,channel] = group_channel(image[:,:,channel], gcount)

    return image

def threshold(channel, thresh=(128,255), thresh_type=cv2.THRESH_BINARY):
    # threshold the supplied channel
    # channel : 2D array of the channel data
    # thresh : 2D tupple of min and max threshold values
    # thresh_type : what type of threshold to apply
    # return : 2D thresholded data
    return cv2.threshold(channel, thresh[0], thresh[1], thresh_type)

def adaptive_gaus_thresh(channel, max_value=255, adapt_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type=cv2.THRESH_BINARY, ksize=3, c=0):
    # uses a threshold technique (gaussian, binary) to threshold the input channel
    # max_value : upper limit to the displayed threshold value
    # adapt_method : threshold method, cv2.ADAPTIVE_THRESH_GAUSSIAN_C or cv2.ADAPTIVE_THRESH_MEAN_C
    # thresh_type : type of threshold, binary, tozero, trunc
    # ksize : kernel size for the nearest neighboure
    # c : constant positive or negative
    # return : the thresholded channel
    return cv2.adaptiveThreshold(channel, max_value, adapt_method, thresh_type, ksize, c)

def adaptive_otsu_thresh(channel, thresh=(0,255), ksize=3, thresh_type=cv2.THRESH_BINARY):
    # Otsu's thresholding after Gaussian filtering
    # channel : 2D input channel data
    # thresh : 2D tupple of min and max threshold values
    # ksize : 2D tupple with the size of the blurring kernel
    # thresh_type : what type of threshold to apply
    # return : 2D thresholded data
    blur = cv2.GaussianBlur(channel, (ksize, ksize), 0)
    return cv2.threshold(blur, thresh[0], thresh[1], thresh_type + cv2.THRESH_OTSU)

def remove_background(img):
    # removes the background from img
    # img : image to have the background removed
    # result : image with background removed
    fgbg = cv2.createBackgroundSubtractorMOG2()
    return fgbg.apply(img)

def washout(img):
    gray = img[:,:,0] #cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = adaptive_otsu_thresh(gray, thresh_type=cv2.THRESH_BINARY_INV)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    img[markers != 1] = [255, 0, 0]

    return img

def norm_color_channel(channel, target, tar_var=10, ksize=(1,1), stride=(1,1)):
    # takes 2D data and makes all the data in the size of the kernel have the
    # mean value of all the data in the that kernel
    # channel : 2D data that will be normalised
    # target : target value
    # tar_var : allowed varience from the target value
    # ksize : size of the kernel
    # stride :
    # return : normalised channel using the mean
    for row in range(0, channel.shape[0], ksize[0]):
        for col in range(0, channel.shape[1], ksize[1]):
            mean = np.median(channel[row:row+ksize[0], col:col+ksize[1]])
            if (mean >= target-tar_var) & (mean <= target+tar_var):
                channel[row:row+ksize[0], col:col+ksize[1]] = target
    return channel

def norm_color_image(img, roi_pts, tar_var=10, ksize=(1,1)):
    # takes 3D image data and makes all the data in the size of the kernel have the
    # mean value of all the data in the that kernel
    # img : 3D image data that will be normalised
    # ksize : size of the kernel
    # return : np array of the transformed image
    h, w = img.shape[:2]

    target = np.median(img[h//2:h*3//4, w//4:w*3//4, 0])
    image1 = norm_color_channel(img[:,:,0], target, tar_var=tar_var, ksize=ksize)
    target = np.median(img[h // 2:h * 3 // 4, w // 4:w * 3 // 4, 1])
    image2 = norm_color_channel(img[:,:,1], target, tar_var=tar_var, ksize=ksize)
    target = np.median(img[h // 2:h * 3 // 4, w // 4:w * 3 // 4, 2])
    image3 = norm_color_channel(img[:,:,2], target, tar_var=tar_var, ksize=ksize)

    return np.dstack((image1, image2, image3)).astype(np.uint8)

def blur(img, ksize=3):
    # blures the channel using gaussian blur
    # img : 2D or 3D array to be blured
    # ksize : size of the blurring kernel
    # return : blured 2D data

    return cv2.medianBlur(img, ksize=ksize)

def blur_gaussian(channel, ksize=3):
    # blurs the channel data using gaussian
    # channel : 2D array of data to be blured
    # ksize : size of the kernel
    # return : blured data
    return cv2.GaussianBlur(channel, (ksize, ksize), 0)

def canny_edge(channel, sigma=0.33):
    # finds the canny edges of a channel
    # channel : 2D channel to find edges
    # sigma : difference from the channel mean to find edges (0 -> 1)
    # return : canny edge detected image

    # make sure that sigma is positive and in bounds 0 to 1
    sigma = np.absolute(sigma)
    if sigma > 1:
        sigma = 1

    # compute the median of the single channel pixel intensities
    v = np.median(channel)
    # apply automatic Canny edge detection using the computed median
    low_threshold = int(max(0, (1 - sigma) * v))
    high_threshold = int(min(255, (1 + sigma) * v))

    # find edges in the image using canny edge detection
    return cv2.Canny(channel, low_threshold, high_threshold)
