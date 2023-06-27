import streamlit as st
import cv2
import numpy as np
from api.cv.similarity_model.model import load_trained_encoder
from api.cv.similarity_model.inference import get_distance, preprocess_img
from api.cv.similarity_model.utils import decode_image_to_bgr
from api.cv.face_detector import detect_faces
from PIL import Image


def prepare_zip_file(main_file, list_of_files):
    # Load encoder
    PATH_TO_WEIGHTS = "api/cv/similarity_model/weights/encoder"
    encoder = load_trained_encoder(PATH_TO_WEIGHTS)
    # Read main picture
    main_img = np.array(Image.open(main_file))
    # Read list of pictures
    list_of_imgs = [np.array(Image.open(file)) for file in list_of_files]
    # Get one face from first picture
    main_face = detect_faces(main_img, max_n_faces=1)
    # If we could not detect any face, use the whole picture as face
    if len(main_face) > 0:
        main_face = main_face[0]
    else:
        main_face = main_img
    # Get faces from the rest of the pictures
    all_pictures_faces = [detect_faces(img, max_n_faces=10) for img in list_of_imgs]
    # Convert to RGB for the encoder
    main_face = cv2.cvtColor(main_face, cv2.COLOR_BGR2RGB)
    converted_faces = []
    for idx, faces in enumerate(all_pictures_faces):
        rgb_faces = []
        for face in faces:
            rgb_faces.append(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        converted_faces.append(rgb_faces)

    # faces = [cv2.cvtColor(face, cv2.COLOR_BGR2RGB) for face in faces]
    # Use this value as a threshold for a match
    threshold = 1.
    # Store here the indexes of the pictures where we have a match
    matches = []
    # Loop
    for idx, faces in enumerate(converted_faces):
        for face in faces:
            distance = get_distance(
                preprocess_img(main_face),
                preprocess_img(face),
                encoder
            )
            if distance < threshold:
                matches.append(idx)
                break
    # Update indexes to download
    matched_pictures = [list_of_files[i] for i in matches]
    if len(matched_pictures) > 0:
        # Prepare zipfile
        from zipfile import ZipFile
        zipObj = ZipFile("results.zip", "w")
        # Add multiple files to the zip
        [zipObj.write(content) for content in matched_pictures]
        zipObj.close()
        st.session_state.zip_file_ready = True
    return


# Store here the path to the zip file
st.session_state.zip_file_ready = False

# Main title
st.title("Find yourself in your pictures")

# Add section
st.subheader("1. Upload your own picture here")

# Upload your own file
main_picture = st.file_uploader(
    label="We need a nice picture of yourself here",
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=False,
    key='main_picture'
)

# Reveal new subsection
if main_picture is not None:
    st.subheader("Upload all your pictures here")

    # Upload your own file
    all_pictures = st.file_uploader(
        label="Upload the pictures you want to search for yourself",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        key='search_pictures'
    )

    if len(all_pictures) >= 1:
        st.button(
            label="Find me!",
            on_click=prepare_zip_file,
            args=(main_picture, all_pictures)
        )

# if st.session_state.zip_file_ready:
#     st.write("-------")
#     with open("results.zip", "rb") as fp:
#         st.download_button(
#             label="Download ZIP",
#             data=fp,
#             file_name="results.zip",
#             mime="application/zip"
#         )