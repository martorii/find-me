import os
# Set tensorflow logger to error mode
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
from api.cv.similarity_model.model import load_trained_encoder
from api.cv.similarity_model.inference import preprocess_img, get_distance
import cv2
from api.cv.face_detector import detect_faces
import shutil
import logging
logging.basicConfig(level=logging.INFO)
import tqdm

# Delete all files from the output folder
[os.remove(f) for f in glob.glob("output/*")]

"""
Process main face
"""
# Get path to main face
all_faces_in_dir = glob.glob("input/*.png") + glob.glob("input/*.jpg") + glob.glob("input/*.jpeg")
path_to_main_face = all_faces_in_dir[0]

# Read image
main_face_img = cv2.imread(path_to_main_face)

# Search for face
main_face = detect_faces(main_face_img, max_n_faces=1)

# If more than one, take the first. If none, take the whole picture.
if len(main_face) >= 1:
    main_face = main_face[0]
else:
    main_face = main_face_img

# Convert to RGB for the encoder
main_face = cv2.cvtColor(main_face, cv2.COLOR_BGR2RGB)

# Preprocess it
main_face = preprocess_img(main_face)

"""
Preprocess the rest of the pictures
"""
# Get paths to all other pictures
paths_to_pictures = glob.glob("input/pictures/*.png") + glob.glob("input/pictures/*.jpg") + glob.glob("input/pictures/*.jpeg")

# Read images
img_list = [cv2.imread(path) for path in paths_to_pictures]

# Search for faces
all_faces = [detect_faces(img, max_n_faces=25) for img in img_list]

# Set threshold
threshold = 1.
matches = []

# Create a new similarity_model instance and load pretrained weights
encoder = load_trained_encoder('api/cv/similarity_model/weights/encoder')

# Loop
for idx, faces in enumerate(tqdm.tqdm(all_faces)):
    for face in faces:
        # Convert to rgb for the encoder
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        distance = get_distance(
            main_face,
            preprocess_img(face_rgb),
            encoder
        )
        if distance < threshold:
            matches.append(idx)
# Select only those pictures that matched
paths_to_pictures = [paths_to_pictures[i] for i in matches]
# Copy to output
[
    shutil.copy(
        path, path.replace("input/pictures", "output")
    ) for path in paths_to_pictures
]
logging.info("Done!")
