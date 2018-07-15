from styx_msgs.msg import TrafficLight

from keras.models import load_model

from .train_classifier import CLASS_LABELS

import numpy as np
import cv2

model_name = "light_classification/tl_classifier_01.hdf5"
input_image_size_for_model = (150, 150)


class TLClassifier(object):
    def __init__(self):
        # somehow, loading the model corrupts the process ...??
        # don't do this for moment as no one has the model
        # self.model = load_model(model_name)
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        resized_image = cv2.resize(image, input_image_size_for_model)
        predicted_label = self.model.predict_classes(np.expand_dims(resized_image, axis=0), batch_size=1)
        return CLASS_LABELS[predicted_label]
