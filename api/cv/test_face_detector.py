import cv2
import random
import matplotlib.pyplot as plt
import sys
import os

# Pick a path from a list randomly
list_of_examples = [
    # 'images/img_mark.jpg',
    '../cv/similarity_model/data/own/face_christina_2.png'
]
i = random.randint(0, len(list_of_examples)-1)
path_to_picture = list_of_examples[i]

# Read image
img = cv2.imread(path_to_picture)

# Add module to sys.path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# Import module from cv
import face_detector

# Detect faces and return cropped faces
faces = face_detector.detect_faces(img)

cv2.imwrite('similarity_model/data/own/face_christina_2_cv2.png', cv2.cvtColor(faces[0], cv2.COLOR_BGR2RGB))

img_rgb = cv2.cvtColor(faces[0], cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20,10))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()


