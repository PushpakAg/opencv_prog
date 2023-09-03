import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.models import Model
import cv2

# Function to generate synthetic cloud images and perform registration
def generate_synthetic_data(num_samples=100):
    # Create synthetic cloud images with random translations
    images = []
    target_images = []
    for _ in range(num_samples):
        cloud_image = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
        h, w, _ = cloud_image.shape
        translation_x = np.random.randint(-50, 50)
        translation_y = np.random.randint(-50, 50)
        M = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
        target_image = cv2.warpAffine(cloud_image, M, (w, h))
        images.append(cloud_image)
        target_images.append(target_image)

    return np.array(images), np.array(target_images)

# Function to build the spatial transform network
def build_spatial_transform_network():
    input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
    spatial_transform_layer = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    spatial_transform_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(spatial_transform_layer)
    spatial_transform_layer = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(spatial_transform_layer)
    spatial_transform_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(spatial_transform_layer)
    spatial_transform_layer = tf.keras.layers.Flatten()(spatial_transform_layer)
    spatial_transform_layer = tf.keras.layers.Dense(64, activation='relu')(spatial_transform_layer)
    theta = tf.keras.layers.Dense(6, activation='linear')(spatial_transform_layer)
    model = Model(inputs=input_layer, outputs=theta)

    return model

# Main function for image registration using spatial transform
def image_registration_with_spatial_transform():
    # Generate synthetic cloud images
    num_samples = 100
    images, target_images = generate_synthetic_data(num_samples=num_samples)

    # Preprocess images for the InceptionResNetV2 model
    images_preprocessed = preprocess_input(images.astype(float))
    target_images_preprocessed = preprocess_input(target_images.astype(float))

    # Load the InceptionResNetV2 CNN model for feature extraction
    cnn_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    cnn_model = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-1].output)

    # Extract features from images using the CNN model
    features = cnn_model.predict(images_preprocessed)
    target_features = cnn_model.predict(target_images_preprocessed)

    # Build the spatial transform network
    stn_model = build_spatial_transform_network()

    # Compile the model with Mean Squared Error loss
    stn_model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the spatial transform network
    stn_model.fit(features, target_features, epochs=10, batch_size=16)

    # Use the trained model to perform image registration
    registered_images = []
    for i in range(num_samples):
        theta = stn_model.predict(features[i:i + 1])
        theta = theta.reshape((2, 3))
        registered_image = cv2.warpAffine(images[i], theta, (224, 224))
        registered_images.append(registered_image)

    return images, target_images, registered_images

# Example usage
if __name__ == "__main__":
    images, target_images, registered_images = image_registration_with_spatial_transform()

    # Display the original images, target images, and registered images
    for i in range(len(images)):
        cv2.imshow('Original Image', images[i])
        cv2.imshow('Target Image', target_images[i])
        cv2.imshow('Registered Image', registered_images[i])
        cv2.waitKey(0)

    cv2.destroyAllWindows()
