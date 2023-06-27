import os
# Set tensorflow logger to error mode
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model import load_trained_encoder, read_img_from_path_as_rgb
import matplotlib.pyplot as plt
from inference import preprocess_img, get_distance
from numpy import round

# Create a new similarity_model instance and load pretrained weights
encoder = load_trained_encoder('weights/encoder')

# Load an anchor, a positive and a negative example
anchor_path = "data/own/face_christina_cv2.png"
positive_path = "data/own/face_christina_2_cv2.png"
negative_path = "data/own/face_andrew.png"

# anchor_path = "data/own/face_mark.png"
# positive_path = "data/own/face_mark_2.png"
# negative_path = "data/own/face_andrew.png"

# Anchor
anchor = read_img_from_path_as_rgb(anchor_path)
preprocessed_anchor = preprocess_img(anchor)

# Positive
positive = read_img_from_path_as_rgb(positive_path)
preprocessed_positive = preprocess_img(positive)

# Negative
negative = read_img_from_path_as_rgb(negative_path)
preprocessed_negative = preprocess_img(negative)

# Plot results
f, axes = plt.subplots(1, 3, figsize=(10, 10))
axes[0].imshow(anchor)
axes[1].imshow(positive)
axes[2].imshow(negative)

# Calculate similarity according to model
anchor_positive_similarity = get_distance(preprocessed_anchor, preprocessed_positive, encoder)
anchor_negative_similarity = get_distance(preprocessed_anchor, preprocessed_negative, encoder)

f.suptitle(
    f"a-p = {round(anchor_positive_similarity, 1)}, a-n = {round(anchor_negative_similarity, 1)}"
)
plt.show()

# Get faces

