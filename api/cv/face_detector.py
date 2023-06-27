import cv2
import numpy as np
from typing import List


def detect_faces(img: np.ndarray, max_n_faces: int = 5) -> List[np.ndarray]:
    """
    Given a path to an image, it returns a list of cropped faces as numpy arrays
    :param img: image in numpy format
    :param max_n_faces: maximum number of faces to be returned
    :return: A list of cropped faces from the image
    """
    # Read image
    # img = cv2.imread(path_to_img)
    # Convert to grayscale so that it works faster
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Load face detector from opencv
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    # Find faces
    faces = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.3, minNeighbors=5, minSize=(35, 35)
    )
    # If we found more faces than the threshold, keep only the first ones
    if len(faces) > max_n_faces:
        faces = faces[:max_n_faces]
    # Crop each face from image and store it into a list
    cropped_faces = []
    for (x, y, w, h) in faces:
        cropped_faces.append(img[y:y + h, x:x + w])
    # Return the desired list of faces
    return cropped_faces
