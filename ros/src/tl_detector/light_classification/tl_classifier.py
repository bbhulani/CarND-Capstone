from styx_msgs.msg import TrafficLight

import tensorflow as tf
import keras
import keras.models
import h5py

from .train_classifier import CLASS_LABELS, COLOR_IMAGE_SIZE, BW_IMAGE_SIZE

import numpy as np
import cv2

import os

model_name = "light_classification/tl_classifier_00.hdf5"


class TLClassifier(object):
    def __init__(self, class_label_to_state_as_int32):

        self.class_label_to_state_as_int32 = class_label_to_state_as_int32

        if os.path.exists(model_name):
            # check that model Keras version is same as local Keras version
            f = h5py.File(model_name, mode='r')
            model_version = f.attrs.get('keras_version')
            keras_version = str(keras.__version__).encode('utf8')

            if model_version != keras_version:
                print('You are using Keras version ', keras_version,
                      ', but the model was built using ', model_version)

            # not sure why this is needed -- workaround for issue described here ... https://github.com/keras-team/keras/issues/6462
            self.graph = tf.get_default_graph()

            self.model = keras.models.load_model(model_name)

            # do this in advance
            self.model._make_predict_function()
        self.model = None

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        with self.graph.as_default():
            if image[2] > 1:
                resized_image = cv2.resize(image, COLOR_IMAGE_SIZE[:2])
            else:
                grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                resized_image = cv2.resize(image, BW_IMAGE_SIZE[:2])
            predicted_label = self.model.predict_classes(
                np.expand_dims(resized_image, axis=0),
                batch_size=1,
                verbose=0
            )

            class_label = CLASS_LABELS[predicted_label]
            class_int = self.class_label_to_state_as_int32[class_label]
            return class_int
