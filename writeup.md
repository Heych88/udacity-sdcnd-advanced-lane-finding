**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration2.jpg "Undistorted"
[image2]: ./output_images/undistorted_calibration2.jpg "Road Transformed"
[image3]: ./test_images/test5.jpg "Binary Example"
[image4]: ./output_images/post_process_image_binary_test5.jpg "Post processed image binary"
[image5]: ./output_images/roi_test2.jpg "ROI Visual""
[image6]: ./output_images/warp_image_test5.jpg "Warped Visual of the binary image in Post processed image binary"
[image7]: ./output_images/line_search_test5.jpg "lane line find Visual""
[image8]: ./output_images/quadratic_lines.png "Quadric line equiverlent of the found lines"
[image9]: ./output_images/straight_lines1.jpg "Final lane overlay with road curveture"
[image10]: ./output_images/undistorted_car.jpg "undistorted road"
[video1]: ./output_images/project_video.mp4 "Video"
[video2]: ./output_images/drivingnight.avi "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file called `main.py` at lines 13 to 19. The code usere the calibration functions in `camera.py` and calls the function `CameraCalibration.get_optimal_calibration()` at lines 146 to 159. This returns the calibration matrix of the distorted camera. This function takes in two arguments, `img` - the checkboard image for calibration, `chess_count` - the number of corners in `img`. The output of the function is saved into the class varibles for later calibration of images.

After performing the above function callibration, I verified the distortion correction to the test image using the `CameraCalibration.undistort_image()` function at lines 132 to 144 in the file `camera.py` and obtained this result:

![alt text][image1] "Input image"
![alt text][image2] "Undistorted Input image"

---
### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image10]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of RGB and HSL color spectrums combined with image thresholding, gradient thresholding and bitwise opperations to generate a binary image of the road surface. (steps performed in `Lane.driving_lane()` in `drivingline.py` at lines 467 through 488).  Here's an example of the input image and the output for this step.

![alt text][image3] Input image
![alt text][image4] Output binary image

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

To perform a perspective transformation, a region of interest (roi) needs to be located by calling the function `Lane.lane_roi()`, which appears in lines 62 through 99 in the file `driveline.py`. This function requires three arguments to be set:
1. `focal_point` must be a tuple in the location of the road vanishing point.
2. `source_pts` must be a list of tuples for the outer roi points. i.e the bottom left and right corners of the image.
3. `roi_height` is the 'region of interest' (roi) cutoff height from the focal_point location. It is a fraction of the distance from the source_pts to the focal point.

![alt text][image5] An example of the roi is below:

Once the roi is located code for my perspective transform includes a function called `Lane. warp_image()`, which appears in lines 121 through 145 in the file `driveline.py` (see the below image transformed on the output binary image).  The `warper()` function takes as inputs an image (`img`), as well as the located roi points as the source points (`src`) and the final destination (`dst`) points which were choosen as the input image size.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image6] Warped image of the 'Post processed image binary'

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To locate the lane lines in the warped image, I applied a histogram window filter called `Lane.lane_lines()`, which appears in lines 442 through 465 in the file `driveline.py`. This function calles on either one of two other function, depending on if the lanes have already been located.
* If it has not already located the lanes, it calls `Lane.find_lane_lines()`, which appears in lines 324 through 404 in the file `driveline.py`.
* If it has already located the lanes, it calls `Lane.refresh_lane_lines()`, which appears in lines 406 through 440 in the file `driveline.py`.

![alt text][image7]

In each of the above functions, the located lines were then put into the 2nd order polynomial function `numpy.polyfit()`. This was performed on both the left and right lane lines and each function fas passed into an average filter called `Filter.moving_average()`, which appears in lines 20 through 36 in the file `filter.py`. The filter averaged the line functions over the past 32 calculations to minimise noise. Below is an output graph of the located lane lines. The right lane line is shown in Blue, Left lane line shown in Orange and the average of both of them is disolayed in Green.

![alt text][image8]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the function `Lane.lane_lines_radius`, at lines 258 through 269 in my code in `driveline.py`

The reult is displayed onto the image by default but can be removed by passing in the argument, `display_curve=False` into the function `lane_pipeline()` located at line 99 in my code in the file `main.py`. See the below image in sections 6 for an example.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 510 through 531 in my code in `driveline.py` in the function `Lane.overlay_lane()`.  Here is an example of my result on a test image:

![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my project video][video1] and heres is a [link to my night driving video][video2]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach taken for the project was to be able to seperate colors, lines and other contrasting objects from one another in the image. I.e by removing the road region and similar colours of the road can remove noise from the input data. This was seen as the best way to minimise road noise and provide a clean crisper data for finding line edges.

Many methods and attempts to remove shadows and unnedded data where attempted, such as adjusting the brightness of the image using any of the below functions:
`camera.py`
* `CameraImage.adjust_normal_hist()` - A histogram kernel brightness adjuster.
* `CameraImage.adjust_image_gamma()` - Adjusts the gamma of the image
* `CameraImage.auto_adjust_gamma()` - Adjust the image gamma bassed of the mean of the image.
* `CameraImage.adjust_channel()` - Adjust a single channel brightness using a histogram kernel approch.
* `CameraImage.adjust_channel_gamma()` - Adjust a singl channels gamma.

It was discovered that the the above brightness adjustment methods were effective but a similar result was also observed by having the correct threshold values when binarising channels. It was decided that due to the added computation required for image brightness adjustment, a binarised channel with the correct threshold performed as good, but much faster.
However, at this stage threshold values need to change when driving on different road surfaces and conditions. At this stage the threshold values are hard coded and should be implemented as an adaptive method for a more portable solution.

issues
* On sharp corners the road lines can leave the region of interest area through the side and not through the desired top. This minimises accuracy of that line due to the smaller amount of data being fitted to the quadratic. This is also a problem when the road goes both uphill (raod lines disappear out both sides of the roi box) or going downhill where the road dips away from the horizon and now the horizon is considered in the roi area adding noise to the system.
In the future, The roi points need to be adaptive. That is the `focus_point` should always track the road horizon, moving in a verticle manner, as well as moving side to side on bends to always keep the road lanes from exiting through the roi top.

* At the moment of implementation, to lane line locations are fixed and should they disappear from a section of the road for a resonable time, the system will lose track of the lines and will not go into a search mode to find the lines.
In the future, dead lines should be detected and a new search should be conducted to locate them again once they appear.

* Lanes are lost and exceptions are thrown if the vehicle changes lanes.
Future handel exceptions and perform a new search outside the roi to locate the new lanes.

* Other vehicles in the line of sight. At the moment if a vehicle is travelling in the same lane close to the camera, the car edges are added to the quadratic lines and cause incorrect lane locations.
Future, track cars and remove their noise impact on the calculations.

As the current code stands. It works on clearly marked roads both during the day and at night time. But further implementations will be required to implement on the above issues.
