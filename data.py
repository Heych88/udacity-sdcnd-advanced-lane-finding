"""
**********************************************************************
*    image and input files path directories
**********************************************************************
"""
a1 = 'exit-ramp.jpg'
a2 = 'straight_lines1.jpg'
a3 = 'straight_lines2.jpg'
a4 = 'test1.jpg'
a5 = 'test2.jpg'
a6 = 'test3.jpg'
a7 = 'test4.jpg'
a8 = 'test5.jpg'
a9 = 'test6.jpg'

v1 = 'challenge_video.mp4'
v2 = 'project_video.mp4'
v3 = 'harder_challenge_video.mp4'

isVideo = False

if isVideo:
    video = v2
    folder_add = ''
    img_add = video
else:
    image = a4
    folder_add = 'test_images/'
    img_add = 'test_images/' + image

# load certain parameters depending on the input image
if img_add == folder_add + a1:
    fph = 50/100    # height of the focal point
    phm = 24        # distance in meters that the lane roi reaches
    roi_height = 89/100
elif img_add == folder_add + a2:
    fph = 58 / 100
    phm = 30
    roi_height = 89 / 100
elif img_add == folder_add + a3:
    fph = 57/100
    phm = 30 #
    roi_height = 89 / 100
elif img_add == folder_add + a4:
    fph = 57/100
    phm = 21 #
    roi_height = 89 / 100
elif img_add == folder_add + a5:
    fph = 57/100
    phm = 30 #
    roi_height = 89 / 100
elif img_add == folder_add + a6:
    fph = 58 / 100
    phm = 27 #
    roi_height = 87 / 100
elif img_add == folder_add + a7:
    fph = 57 / 100
    phm = 24
    roi_height = 89 / 100
elif img_add == folder_add + a8:
    fph = 57/100
    phm = 24 #
    roi_height = 89 / 100
elif img_add == folder_add + a9:
    fph = 57/100
    phm = 24 #
    roi_height = 87 / 100
elif img_add == v1:
    fph = 60/100
    phm = 24
    roi_height = 87 / 100
elif img_add == v2:
    fph = 57/100
    phm = 24
    roi_height = 87 / 100
elif img_add == v3:
    fph = 30 / 100
    phm = 26
    roi_height = 87 / 100

def trim_data(img):

    if isVideo:
        if img_add == v1:
            h, w = img.shape[:2]
            img = img[0:h-50, :]
        elif img_add == v2:
            h, w = img.shape[:2]
            img = img[0:h-70, :]
    else:
        # read in the image to the program
        if img_add == a3 or img_add == a4 or img_add == a5:
            h, w = img.shape[:2]
            img = img[0:h-50, :]
    return img