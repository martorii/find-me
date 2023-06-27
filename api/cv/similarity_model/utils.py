import os
import random
import cv2
from typing import Tuple, List
import numpy as np


def split_dataset(directory: str, split: float = 0.9) -> Tuple[dict[str, int], dict[str, int]]:
    """
    Given a directory, split it into train and test data
    :param directory: folder containing subfolders with data
    :param split: how much % should be used for training?
    :return: Two dictionaries containing train and test mapping
    """
    # Get all the folders in the given directory
    folders = [f.name for f in os.scandir(directory) if f.is_dir()]
    # How many folders are going to be used for training?
    num_train = int(len(folders) * split)
    # Shuffle them
    random.shuffle(folders)
    # Create dictionaries to store train and test data
    train_list, test_list = {}, {}
    # Creating train data
    for folder in folders[:num_train]:
        num_files = len(os.listdir(os.path.join(directory, folder)))
        train_list[folder] = num_files
    # Creating test data
    for folder in folders[num_train:]:
        num_files = len(os.listdir(os.path.join(directory, folder)))
        test_list[folder] = num_files
    return train_list, test_list


def create_triplets(directory: str, folder_list: dict[str, int], max_files: int = 10) -> List[Tuple[str, str, str]]:
    """
    Create a list with triplets to be used for training
    :param directory:
    :param folder_list:
    :param max_files:
    :return:
    """
    # Store triplets here
    triplets = []
    # Get a list with all folders
    folders = list(folder_list.keys())
    # Iterate over all folders
    for folder in folders:
        path = os.path.join(directory, folder)
        files = list(os.listdir(path))[:max_files]
        num_files = len(files)
        # Iterate over files in each folder
        for i in range(num_files - 1):
            for j in range(i + 1, num_files):
                # Create anchor, positive and negative example
                anchor = os.path.join(path, f"{i}.jpg")
                positive = os.path.join(path, f"{j}.jpg")
                neg_folder = folder
                while neg_folder == folder:
                    neg_folder = random.choice(folders)
                neg_file = random.randint(0, folder_list[neg_folder] - 1)
                negative = os.path.join(directory, neg_folder, f"{neg_file}.jpg")
                # Append to triplets list
                triplets.append((anchor, positive, negative))
    # Shuffle for randomness
    random.shuffle(triplets)
    return triplets


def decode_image_to_bgr(contents: bytes) -> np.ndarray:
    """
    Given an fastapi File, convert it to an numpy image
    :param contents: file containing the image
    :return: numpy image
    """
    # Decode bytes
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    # Convert to numpy array
    img = np.array(img)
    return img
