# Analyz-Material
Analyz Material
# Code for Material Recognition using OpenCV and Machine Learning
import cv2
import numpy as np
from sklearn.cluster import KMeans

def recognize_materials(image_path):
    # Read the input image
    image = cv2.imread(image_path)

    # Resize the image for better performance (optional)
    image = cv2.resize(image, (400, 400))

    # Flatten the image into a list of RGB values
    pixels = image.reshape((-1, 3))

    # Use KMeans clustering to find dominant colors
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(pixels)

    # Get the dominant colors
    dominant_colors = kmeans.cluster_centers_.astype(int)

    # Display the dominant colors
    for color in dominant_colors:
        color_patch = np.full((100, 100, 3), color, dtype=np.uint8)
        cv2.imshow('Dominant Color', color_patch)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

# Example usage
image_path = "path/to/your/image.jpg"
recognize_materials(image_path)
