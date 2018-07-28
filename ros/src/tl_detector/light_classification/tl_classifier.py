import os

from styx_msgs.msg import TrafficLight

import tensorflow as tf
import numpy as np
import cv2

# Configure whether ot use the whole image classifier (implemented with keras)
# or the semantic segmentation network (implemented with just tensorflow)
USE_KERAS_CLASSIFIER = True

if USE_KERAS_CLASSIFIER:
    # Use simple whole image classifier

    import keras
    import keras.models
    import h5py

    from .train_classifier import CLASS_LABELS, COLOR_IMAGE_SIZE, BW_IMAGE_SIZE

    IMAGE_SIZE = COLOR_IMAGE_SIZE

    model_name = "light_classification/tl_classifier_00.hdf5"

    class TLClassfieir(object):
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
                if IMAGE_SIZE[2] > 1:
                    resized_image = cv2.resize(image, IMAGE_SIZE[:2])
                else:
                    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    resized_image = cv2.resize(image, IMAGE_SIZE[:2])
                predicted_label = self.model.predict_classes(
                    np.expand_dims(resized_image, axis=0),
                    batch_size=1,
                    verbose=0
                )

                class_label = CLASS_LABELS[predicted_label]
                class_int = self.class_label_to_state_as_int32[class_label]
                return class_int

else:
    # Use semantic segmentation classifier
    IMAGE_SIZE = (256, 256)  # this can be any multiple of 32 for this model
    model_name = "./trafficlight-segmenter/model"
    from train_segmenter import restore_model

    CLASS_LABELS = [
        "unknown",
        "red",
        "green",
        "yellow"
    ]

    from .dataset import reverse_one_hot

    class TLClassfieir(object):
        def __init__(self, class_label_to_state_as_int32):

            self.class_label_to_state_as_int32 = class_label_to_state_as_int32
            self.session = tf.Session()

            if os.path.exists(model_name):
                self.logits, self.keep_prob, self.input_image, self.predicted_label_probabilities = restore_model(self.session, model_name)

            self.red_light_threshold = 20
            self.green_light_threshold = 20
            self.yellow_light_threshold = 20

        def get_classification(self, images):
            """Determines the color of the traffic light in the image

            Args:
                image (cv::Mat): image containing the traffic light

            Returns:
                int: ID of traffic light color (specified in styx_msgs/TrafficLight)

            """
            if not isinstance(images, list):
                images = [images]

            resized_images = []
            for img in images:
                resized_image = cv2.resize(img, IMAGE_SIZE[:2])
                resized_image = np.expand_dims(resized_image, axis=0)
                resized_images.append(resized_image)

            sess = self.session

            feed_dict = {
                self.input_image: resized_images,
                self.keep_prob: 1.0
            }

            label_probabilities = sess.run([
                self.predicted_label_probabilities
            ],
                feed_dict=feed_dict
            )

            # minimal viable concept, should do something better to go from
            # segmentation map to traffic light prediction ...
            labeled_image = reverse_one_hot(label_probabilities)
            label_count = np.sum(labeled_image, axis=3)

            if label_count[1] > self.red_light_threshold:
                class_label = 1
            elif label_count[2] > self.green_light_threshold and self.green_light_threshold < self.red_light_threshold:
                class_label = 2
            elif label_count[2] > self.yellow_light_threshold and self.yellow_light_threshold < self.red_light_threshold and self.yellow_light_threshold < self.green_light_threshold:
                class_label = 3

            return self.class_label_to_state_as_int32[class_label]
