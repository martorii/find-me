"""
Thanks to https://www.kaggle.com/code/erikmartorilpez/face-recognition-siamese-w-triplet-loss/edit
"""
import time
import os
# Set tensorflow logger to error mode
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import split_dataset, create_triplets
from metrics import plot_metrics
from keras.optimizers import Adam
import numpy as np


# Path where the train/test data is stored
path_to_data = "data/face-recognition-dataset/extracted-faces/"

# Get train and test data
train_list, test_list = split_dataset(path_to_data, split=0.9)
print("Length of training list:", len(train_list))
print("Length of testing list :", len(test_list))

# Create triplets for train/test
train_triplets = create_triplets(path_to_data, train_list)
test_triplets = create_triplets(path_to_data, test_list)

# Create model
from model import get_siamese_network, SiameseModel, get_batch, extract_encoder

siamese_network = get_siamese_network()
siamese_model = SiameseModel(siamese_network)
optimizer = Adam(learning_rate=1e-3, epsilon=1e-01)
siamese_model.compile(optimizer=optimizer)

# Train
from metrics import get_metrics

save_all = False
EPOCHS = 2
BATCH_SIZE = 64

max_acc = 0
train_loss = []
test_metrics = []

for epoch in range(1, EPOCHS + 1):
    t = time.time()
    # Training the model on train data
    epoch_loss = []
    for data in get_batch(train_triplets, batch_size=BATCH_SIZE):
        loss = siamese_model.train_on_batch(data)
        epoch_loss.append(loss)
    epoch_loss = sum(epoch_loss) / len(epoch_loss)
    train_loss.append(epoch_loss)

    print(f"\nEPOCH: {epoch} \t (Epoch done in {int(time.time() - t)} sec)")
    print(f"Loss on train    = {epoch_loss:.5f}")

    # Testing the model on test data
    metric = get_metrics(test_triplets, BATCH_SIZE, siamese_model)
    test_metrics.append(metric)
    accuracy = metric[0]

    # Saving the model weights
    if save_all or accuracy >= max_acc:
        siamese_model.save_weights("weights/siamese_model.ckpt")
        max_acc = accuracy

# Plot metrics
test_metrics = np.array(test_metrics)
plot_metrics(train_loss, test_metrics)

# Extract encoder
encoder = extract_encoder(siamese_model)
encoder.save_weights("weights/encoder")


def classify_images(face_list1, face_list2, threshold=1.3):
    # Getting the encodings for the passed faces
    tensor1 = encoder.predict(face_list1)
    tensor2 = encoder.predict(face_list2)

    distance = np.sum(np.square(tensor1 - tensor2), axis=-1)
    prediction = np.where(distance <= threshold, 0, 1)
    return prediction

pos_list = np.array([])
neg_list = np.array([])

for data in get_batch(test_triplets, batch_size=256):
    a, p, n = data
    pos_list = np.append(pos_list, classify_images(a, p))
    neg_list = np.append(neg_list, classify_images(a, n))
    break

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
def ModelMetrics(pos_list, neg_list):
    true = np.array([0] * len(pos_list) + [1] * len(neg_list))
    pred = np.append(pos_list, neg_list)

    # Compute and print the accuracy
    print(f"\nAccuracy of model: {accuracy_score(true, pred)}\n")

    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(true, pred)

    categories = ['Similar', 'Different']
    names = ['True Similar', 'False Similar', 'False Different', 'True Different']
    percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(names, percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='',
                xticklabels=categories, yticklabels=categories)

    plt.xlabel("Predicted", fontdict={'size': 14}, labelpad=10)
    plt.ylabel("Actual", fontdict={'size': 14}, labelpad=10)
    plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)
    plt.show()