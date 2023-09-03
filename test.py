import cv2
import numpy as np
reference_img = cv2.imread("D:/maincloudref.jpg")
cloud_img = cv2.imread("D:/warpedcloud.jpg")
reference_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
cloud_gray = cv2.cvtColor(cloud_img, cv2.COLOR_BGR2GRAY)
feature_detector = cv2.SIFT_create()  # You can replace SIFT with SURF or ORB if desired
keypoints_ref, descriptors_ref = feature_detector.detectAndCompute(reference_gray, None)
keypoints_cloud, descriptors_cloud = feature_detector.detectAndCompute(cloud_gray, None)
bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf_matcher.match(descriptors_ref, descriptors_cloud)
matches = sorted(matches, key=lambda x: x.distance)
MIN_MATCH_COUNT = 10  # Adjust this value based on your requirements
if len(matches) > MIN_MATCH_COUNT:
    src_pts = np.float32([keypoints_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_cloud[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Calculate the transformation matrix using RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Apply the transformation to the reference image
    registered_img = cv2.warpPerspective(reference_img, M, (cloud_img.shape[1], cloud_img.shape[0]))

    # Visualize the registered image
    cv2.imshow("Registered Image", registered_img)
    cv2.waitKey(0)
else:
    print("Not enough matches found - image registration failed.")
