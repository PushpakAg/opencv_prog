import cv2 
import numpy as np

img1 = cv2.imread("D:/satellite_img/Screenshot_20230801-071027_Earth.jpg")
img2 = cv2.imread("D:/satellite_img/Screenshot_20230801-071100_Earth.jpg")

im1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
im2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(50)

kp1, des1 = orb.detectAndCompute(im1,None)
kp2, des2 = orb.detectAndCompute(im2,None)


matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

matches = matcher.match(des1,des2, None)

pt1= np.zeros((len(matches),2), dtype = np.float32)
pt2= np.zeros((len(matches),2), dtype = np.float32)

for i,match in enumerate(matches):
    pt1[i,:] = kp1[match.queryIdx].pt
    pt2[i, :] = kp2[match.trainIdx].pt

h, mask = cv2.findHomography(pt1, pt2, cv2.RANSAC)

height, width= im2.shape
im1Reg = cv2.warpPerspective(img1, h ,(width,height))

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[0:],None)
cv2.imshow("key matches",img3)
cv2.imshow("reg img", im1Reg)    
cv2.waitKey(0)