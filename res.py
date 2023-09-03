import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.models import Model

import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Function to compute SIFT keypoints and descriptors for an image
def compute_sift(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray_image, None)
    return kp, des

# Function to match SIFT keypoints between two images
def match_sift_features(des1, des2, ratio=0.75):
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    matches = []
    for m1, m2 in raw_matches:
        if m1.distance < ratio * m2.distance:
            matches.append(m1)
    return matches

# Function to perform image registration using CNN features and SIFT
def image_registration_with_cnn_and_sift(reference_image_path, target_image_path):
    # Load the InceptionResNetV2 CNN model for feature extraction
    cnn_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    cnn_model = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-1].output)

    # Load reference and target images
    reference_image = cv2.imread(reference_image_path)
    target_image = cv2.imread(target_image_path)

    # Resize images to 224x224 as InceptionResNetV2 expects this input size
    reference_image = cv2.resize(reference_image, (224, 224))
    target_image = cv2.resize(target_image, (224, 224))

    # Preprocess images for the CNN model
    reference_image_preprocessed = preprocess_input(reference_image)
    target_image_preprocessed = preprocess_input(target_image)

    # Extract features from reference and target images using CNN
    reference_features = cnn_model.predict(np.expand_dims(reference_image_preprocessed, axis=0)).flatten()
    target_features = cnn_model.predict(np.expand_dims(target_image_preprocessed, axis=0)).flatten()

    # Compute SIFT keypoints and descriptors for reference and target images
    reference_kp, reference_des = compute_sift(reference_image)
    target_kp, target_des = compute_sift(target_image)

    # Match SIFT keypoints between reference and target images
    matches = match_sift_features(reference_des, target_des)

    # Get matched keypoints coordinates
    reference_matched_points = np.float32([reference_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    target_matched_points = np.float32([target_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find the affine transformation matrix using matched keypoints
    M, _ = cv2.estimateAffinePartial2D(reference_matched_points, target_matched_points)

    # Warp the target image using the affine transformation matrix
    registered_image = cv2.warpAffine(target_image, M, (reference_image.shape[1], reference_image.shape[0]))

    return reference_image, target_image, registered_image

# Example usage
if __name__ == "__main__":
    reference_image_path = "D:/satellite_img/Screenshot_20230801-071027_Earth.jpg" # Replace with the path to your reference image
    target_image_path = "D:/satellite_img/Screenshot_20230801-071100_Earth.jpg"  # Replace with the path to your target image

    reference_image, target_image, registered_image = image_registration_with_cnn_and_sift(
        reference_image_path, target_image_path
    )

    # Display the images
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.title("Reference Image")
    plt.imshow(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(132)
    plt.title("Target Image")
    plt.imshow(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(133)
    plt.title("Registered Image")
    plt.imshow(cv2.cvtColor(registered_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.show()
