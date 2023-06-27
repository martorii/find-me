from keras.applications.xception import preprocess_input
import numpy as np
from tensorflow import keras
import cv2


def preprocess_img(img: np.ndarray) -> np.ndarray:
    """
    Use xception preprocessor and add extra batch dimension
    :param img: img to be preprocessed
    :return: preprocessed img
    """
    # Resize if needed
    if img.shape[0:2] != (128, 128):
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    # Add batch dimension if missing
    if len(img.shape) < 4:
        img = np.expand_dims(img, axis=0)
    # Preprocess xception
    img = preprocess_input(img)
    return img


def get_distance(img_1: np.ndarray, img_2: np.ndarray, encoder: keras.Sequential) -> float:
    """
    Calculate L2 distance between two images
    :param img_1:
    :param img_2:
    :param encoder: model to be used for encoding
    :return: L2-distance between both images
    """
    encoded_img_1 = encoder.predict(img_1, verbose=0)
    encoded_img_2 = encoder.predict(img_2, verbose=0)
    distance = np.sum(np.square(encoded_img_1 - encoded_img_2), axis=-1)[0]
    return distance
