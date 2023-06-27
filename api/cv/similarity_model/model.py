import numpy as np
import tensorflow as tf
import cv2
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers import BatchNormalization, Layer
from keras.metrics import Mean
from keras import Input, Sequential, Model
from keras.applications import Xception
from keras.applications.xception import preprocess_input

from typing import List

PATH_TO_WEIGHTS = "./weights/encoder"


def read_img_from_path_as_rgb(path: str):
    """
    Read an image as a numpy ndarray
    :param path: path to the image
    :return: the image as numpy format
    """
    # Read from path
    img = cv2.imread(path)
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert to numpy array
    img = np.array(img)
    return img

def extract_encoder(model):
    encoder = get_encoder((128, 128, 3))
    i=0
    for e_layer in model.layers[0].layers[3].layers:
        layer_weight = e_layer.get_weights()
        encoder.layers[i].set_weights(layer_weight)
        i+=1
    return encoder


def get_batch(triplet_list: List, batch_size: int = 256, preprocess: bool = True):
    # Get number of batch steps
    batch_steps = len(triplet_list) // batch_size
    # Iterate for each batch_step
    for i in range(batch_steps + 1):
        # Store anchors, positive and negative for each batch here
        anchor = []
        positive = []
        negative = []
        # Set index
        j = i * batch_size
        # Iterate for a given batch size
        while j < (i + 1) * batch_size and j < len(triplet_list):
            a, p, n = triplet_list[j]
            anchor.append(read_img_from_path_as_rgb(a))
            positive.append(read_img_from_path_as_rgb(p))
            negative.append(read_img_from_path_as_rgb(n))
            j += 1
        # Convert to numpy array
        anchor = np.array(anchor)
        positive = np.array(positive)
        negative = np.array(negative)
        # Preprocess according to the xception pipeline
        if preprocess:
           anchor = preprocess_input(anchor)
           positive = preprocess_input(positive)
           negative = preprocess_input(negative)

        yield ([anchor, positive, negative])


def get_encoder(input_shape):
    """ Returns the image encoding model """

    pretrained_model = Xception(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False,
        pooling='avg',
    )

    for i in range(len(pretrained_model.layers) - 27):
        pretrained_model.layers[i].trainable = False

    encode_model = Sequential([
        pretrained_model,
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(256, activation="relu"),
        Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ], name="Encode_Model")
    return encode_model


class DistanceLayer(Layer):
    # A layer to compute ‖f(A) - f(P)‖² and ‖f(A) - f(N)‖²
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


def get_siamese_network(input_shape=(128, 128, 3)):
    encoder = get_encoder(input_shape)

    # Input Layers for the images
    anchor_input = Input(input_shape, name="Anchor_Input")
    positive_input = Input(input_shape, name="Positive_Input")
    negative_input = Input(input_shape, name="Negative_Input")

    ## Generate the encodings (feature vectors) for the images
    encoded_a = encoder(anchor_input)
    encoded_p = encoder(positive_input)
    encoded_n = encoder(negative_input)

    # A layer to compute ‖f(A) - f(P)‖² and ‖f(A) - f(N)‖²
    distances = DistanceLayer()(
        encoder(anchor_input),
        encoder(positive_input),
        encoder(negative_input)
    )

    # Creating the Model
    siamese_network = Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=distances,
        name="Siamese_Network"
    )
    return siamese_network


class SiameseModel(Model):
    # Builds a Siamese model based on a base-model
    def __init__(self, siamese_network, margin=1.0):
        super(SiameseModel, self).__init__()

        self.margin = margin
        self.siamese_network = siamese_network
        self.loss_tracker = Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape get the gradients when we compute loss, and uses them to update the weights
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.siamese_network.trainable_weights))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # Get the two distances from the network, then compute the triplet loss
        ap_distance, an_distance = self.siamese_network(data)
        loss = tf.maximum(ap_distance - an_distance + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics so the reset_states() can be called automatically.
        return [self.loss_tracker]


def load_trained_encoder(path: str):
    """
    It returns the pretrained encoder
    :return: encoder
    """
    model = SiameseModel(get_siamese_network())
    encoder = extract_encoder(model)
    encoder.load_weights(path)
    return encoder

