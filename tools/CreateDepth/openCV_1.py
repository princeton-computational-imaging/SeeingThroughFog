import cv2 as cv

# Load the left and right images in gray scale
imgL = cv.imread('/Users/Hahner/Downloads/000001/left.png', 0)
imgR = cv.imread('/Users/Hahner/Downloads/000001/right.png', 0)

numDisparities = 64

stereo = cv.StereoBM_create(numDisparities=numDisparities, blockSize=5)

disparity = stereo.compute(imgL,imgR)

cv.imshow('left', imgL)
cv.imshow('disparity', disparity/numDisparities)
cv.waitKey()