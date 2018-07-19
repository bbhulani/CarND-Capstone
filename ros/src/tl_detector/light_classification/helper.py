import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import time
import tensorflow as tf
from glob import glob

import fnmatch
import cv2
import operator
import os

from sklearn.model_selection import train_test_split


def gen_batch_function(dataset_parameters):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """

    training_images = dataset_parameters.training_images()

    random.shuffle(training_images)

    training_image_paths, validation_image_paths = train_test_split(training_images)

    def make_get_batches_fn(image_paths):
        def get_batches_fn(batch_size):
            """
            Create batches of training data
            :param batch_size: Batch Size
            :return: Batches of training data
            """
            for batch_i in range(0, len(image_paths), batch_size):
                images = []
                gt_images = []
                for image_file in image_paths[batch_i:batch_i+batch_size]:
                    gt_image_file = dataset_parameters.groundtruth_image_from_source_image(image_file)

                    image = scipy.misc.imresize(
                        scipy.misc.imread(image_file), dataset_parameters.image_shape)
                    groundtruth_image = scipy.misc.imresize(
                        scipy.misc.imread(gt_image_file), dataset_parameters.image_shape)

                    onehot_groundtruth = one_hot_it(groundtruth_image,
                                                    dataset_parameters.MASK_CLASSES)

                    images.append(image)
                    gt_images.append(onehot_groundtruth)

                yield np.array(images), np.array(gt_images)

        return get_batches_fn

    training_generator = make_get_batches_fn(training_image_paths)
    validation_generator = make_get_batches_fn(validation_image_paths)

    training_generator.num_samples = len(training_image_paths)
    validation_generator.num_samples = len(validation_image_paths)
    return training_generator, validation_generator


def gen_test_output(dataset_parameters, sess, logits, keep_prob, image_placeholder):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_placeholder: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """

    test_images = dataset_parameters.test_images()

    for image_file in test_images:
        image = scipy.misc.imresize(scipy.misc.imread(image_file), dataset_parameters.image_shape)

        feed_dict = {
            keep_prob: 1.0,
            image_placeholder: [image]
        }

        image_softmax = sess.run(
            [tf.nn.softmax(logits)],
            feed_dict=feed_dict
        )[0][0]

        print("image_softmax.shape", image_softmax.shape)

        print(image_softmax)

        image_with_pixels_classified = reverse_one_hot(image_softmax)
        print("image_with_pixels_classified.shape", image_with_pixels_classified.shape)

        image_class_overlay = colour_code_segmentation(image_with_pixels_classified, dataset_parameters.MASK_CLASSES, np.uint8)

        print(image.shape, image.dtype, image_class_overlay.shape, image_class_overlay.dtype)

        alpha = 0.5
        labeled_image = cv2.addWeighted(image_class_overlay, alpha, image, 1 - alpha, 0)

        yield image_file, np.array(labeled_image)


def save_inference_samples(dataset_parameters,
                           sess,
                           logits,
                           keep_prob,
                           input_image):
    # Make folder for output
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)
    # os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    model_outputs = gen_test_output(
        dataset_parameters,
        sess,
        logits,
        keep_prob,
        input_image
    )

    for path, image in model_outputs:
        target_file = dataset_parameters.inferenced_image_from_source_image(path)
        if not os.path.exists(os.path.dirname(target_file)):
            os.makedirs(os.path.dirname(target_file))
        print("Saving file {0}".format(os.path.abspath(target_file)))
        scipy.misc.imsave(target_file, image)


def one_hot_it(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    # st = time.time()
    # w = label.shape[0]
    # h = label.shape[1]
    # num_classes = len(class_dict)
    # x = np.zeros([w,h,num_classes])
    # unique_labels = sortedlist((class_dict.values()))
    # for i in range(0, w):
    #     for j in range(0, h):
    #         index = unique_labels.index(list(label[i][j][:]))
    #         x[i,j,index]=1
    # print("Time 1 = ", time.time() - st)

    # st = time.time()
    # https://stackoverflow.com/questions/46903885/map-rgb-semantic-maps-to-one-hot-encodings-and-vice-versa-in-tensorflow
    # https://stackoverflow.com/questions/14859458/how-to-check-if-all-values-in-the-columns-of-a-numpy-matrix-are-the-same
    semantic_map = []
    for colour in label_values:
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    # print("Time 2 = ", time.time() - st)

    return semantic_map


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,1])

    # for i in range(0, w):
    #     for j in range(0, h):
    #         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
    #         x[i, j] = index

    x = np.argmax(image, axis=-1)
    return x


def colour_code_segmentation(image, label_values, array_type):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """

    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,3])
    # colour_codes = label_values
    # for i in range(0, w):
    #     for j in range(0, h):
    #         x[i, j, :] = colour_codes[int(image[i, j])]

    colour_codes = array_type(label_values)
    x = colour_codes[image.astype(int)]

    return x


def recursive_glob(directory, pattern):
    matches = []
    for root, _, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches


class DatasetParameters:
    model_name = None

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def model_dir(self):
        return os.path.join(self.data_dir, self.model_name)

    def model_savepath(self):
        return os.path.join(self.data_dir, self.model_name, "model")


class TrafficLightParameters(DatasetParameters):
    model_name = "trafficlight-segmenter"
    num_classes = 4
    image_shape = (256, 320)

    MASK_CLASSES = (
        (255, 0, 0),
        (0, 255, 0),
        (255, 255, 0),
        (0, 0, 0)
    )

    MASK_CLASSES_WITH_TRANSPARENCY = (
        (255, 0, 0, 127),
        (0, 255, 0, 127),
        (255, 255, 0, 127),
        (0, 0, 0, 127)
    )

    # MASK_COLORS = {
    #     "red": (255, 0, 0),
    #     "green": (0, 255, 0),
    #     "yellow": (255, 255, 0),
    #     "unknown": (0, 0, 0)
    # }

    def training_images(self):
        dataset_dir = os.path.join(self.data_dir, "bosch_train")
        all_image_paths = recursive_glob(dataset_dir, '*.png')
        return [i for i in all_image_paths if not 'groundtruth' in i]

    def test_images(self):
        return self.training_images()

    def groundtruth_image_from_source_image(self, image_path):
        return os.path.join(os.path.dirname(image_path), "groundtruth_" + os.path.basename(image_path))

    def inferenced_image_from_source_image(self, image_path):
        return os.path.join(self.data_dir, "test_tl", os.path.basename(image_path))


class RoadParameters(DatasetParameters):
    model_name = "road-segmenter"
    num_classes = 3
    image_shape = (256, 320)

    MASK_CLASSES = (
        (0, 0, 0),
        (255, 0, 0),
        (255, 0, 255)
    )

    MASK_CLASSES_WITH_TRANSPARENCY = (
        (0, 0, 0, 127),
        (255, 0, 0, 127),
        (255, 0, 255, 127)
    )

    def training_images(self):
        dataset_dir = os.path.join(self.data_dir, "data_road", "training", "image_2")
        all_image_paths = recursive_glob(dataset_dir, '*.png')
        return all_image_paths

    def test_images(self):
        return self.training_images()

    def groundtruth_image_from_source_image(self, image_path):
        image_filename = os.path.basename(image_path)

        if 'um_' in image_filename:
            image_filename = "um_road_" + image_filename.split('_')[-1]
        elif 'umm_' in image_filename:
            image_filename = "umm_road_" + image_filename.split('_')[-1]
        elif 'uu_' in image_filename:
            image_filename = "uu_road_" + image_filename.split('_')[-1]

        return os.path.join(self.data_dir, "data_road", "training", "gt_image_2", image_filename)

    def inferenced_image_from_source_image(self, image_path):
        return os.path.join(self.data_dir, "test_road", os.path.basename(image_path))
