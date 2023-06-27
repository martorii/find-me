import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import tempfile
import shutil
import tarfile
import os
import json
# Set tensorflow logger to error mode
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from cv.similarity_model.model import load_trained_encoder
from cv.similarity_model.inference import get_distance, preprocess_img
from cv.similarity_model.utils import decode_image_to_bgr
from cv.face_detector import detect_faces
from typing import List
import logging
logging.basicConfig(level=logging.WARNING)

# It starts the API
app = FastAPI()

# Load encoder model
logging.info("Loading encoder model.")
PATH_TO_WEIGHTS = "cv/similarity_model/weights/encoder"
encoder = load_trained_encoder(PATH_TO_WEIGHTS)
logging.info("Encoder model successfully loaded.")


# Define and endpoint to check whether the API is running
@app.get("/health")
async def health():
    return "App is up and running!"


@app.get("/get_face_distance")
async def get_face_distance(files: List[UploadFile] = File(...)):
    """
    Calculate similarity score between two images
    :param files: files containing first images
    :return:
    """
    # Get contents from files
    contents = [await file.read() for file in files]
    # Convert image to cv2 format
    imgs = [decode_image_to_bgr(content) for content in contents]
    # Preprocess
    imgs = [preprocess_img(img) for img in imgs]
    # Calculate similarity
    distance = get_distance(imgs[0], imgs[1], encoder)
    # Return distance
    return json.dumps(
        {'distance': str(distance)},
        indent=4
    )


@app.get("/find_face_in_picture")
async def find_face_in_picture(files: List[UploadFile] = File(...)):
    """
    Given a picture containing only one face and one picture with one or more faces,
    returns a score with a confidence that the first face is in the second picture.
    :param files: Files containing the one-face pic and the multiple-face pic
    :return: confidence that the first face is in the second picture
    """
    # Get contents from files
    contents = [await file.read() for file in files]
    # Convert image to cv2 format
    imgs = [decode_image_to_bgr(content) for content in contents]
    # Get one face from first picture
    main_face = detect_faces(imgs[0], max_n_faces=1)
    # If we could not detect any face, use the whole picture as face
    if len(main_face) > 0:
        main_face = main_face[0]
    else:
        main_face = imgs[0]
    # Get faces from second picture
    faces = detect_faces(imgs[1], max_n_faces=10)
    assert len(faces) >= 1, "No faces were detected in the second picture!"
    # Convert to RGB for the encoder
    main_face = cv2.cvtColor(main_face, cv2.COLOR_BGR2RGB)
    faces = [cv2.cvtColor(face, cv2.COLOR_BGR2RGB) for face in faces]
    # Use this value as a threshold for a match
    threshold = 1.
    # Store all the values here
    distances = []
    # If we have a match, we can exit the loop
    match = False
    for face in faces:
        distance = get_distance(
            preprocess_img(main_face),
            preprocess_img(face),
            encoder
        )
        distances.append(distance)
        if distance < threshold:
            match = True
            break
    return json.dumps(
        {
            'min_distance': str(np.min(distances)),
            'match': match
        },
        indent=4
    )


@app.get("/filter_pictures_with_face")
async def filter_pictures_with_face(files: List[UploadFile] = File(...)):
    """
    Given a picture containing only one face and N pictures with one or more faces,
    returns all the pictures containing the first face.
    :param files: Files containing the one-face pic (first) and the pictures to check (after).
    :return: the pictures that contain the first face in them
    """
    # Get contents from files
    contents = [await file.read() for file in files]
    # Convert image to cv2 format
    imgs = [decode_image_to_bgr(content) for content in contents]
    # Get one face from first picture
    main_face = detect_faces(imgs[0], max_n_faces=1)
    # If we could not detect any face, use the whole picture as face
    if len(main_face) > 0:
        main_face = main_face[0]
    else:
        main_face = imgs[0]
    # Convert to RGB for the encoder
    main_face = cv2.cvtColor(main_face, cv2.COLOR_BGR2RGB)
    # Get faces from the pictures to check
    mapping = {}
    for idx, img in enumerate(imgs[1:]):
        # Detect faces in picture
        faces = detect_faces(img, max_n_faces=10)
        # Convert them into RGB
        faces = [cv2.cvtColor(face, cv2.COLOR_BGR2RGB) for face in faces]
        # Save them in the mapping (add + one since the first is the main face)
        mapping[idx+1] = faces
    # Use this value as a threshold for a match
    threshold = 1.
    # Store here the indexes of the pictures where we have a match
    matches = []
    # Loop
    for idx, faces in mapping.items():
        for face in faces:
            distance = get_distance(
                preprocess_img(main_face),
                preprocess_img(face),
                encoder
            )
            if distance < threshold:
                matches.append(idx)
                break
    # Delete all the pictures where the face is missing
    matched_contents = [contents[i] for i in matches]
    # Create a tempdir to store the files
    with tempfile.TemporaryDirectory() as temp_dir:
        file_name = "results"
        tar = tarfile.open(os.path.join(file_name + ".tar.gz"), 'w:gz')
        for idx, content in enumerate(matched_contents):
            temp_file = tempfile.TemporaryFile(dir=temp_dir)
            temp_file.write(content)
            tar.add(
                os.path.join(str(temp_file.name)), arcname=f"match_{idx}"
            )
        # shutil.make_archive(
        #     base_name=os.path.join(temp_dir, file_name),
        #     format='zip',
        #     root_dir=temp_dir,
        #     base_dir=temp_dir
        # )
        tar.close()
        return FileResponse(
            path=os.path.join(file_name + ".tar.gz")
        )





