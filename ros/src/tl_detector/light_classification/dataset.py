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
from scipy.ndimage.measurements import find_objects
import yaml


def imread_rgb(filename):
    """Read image file from filename and return rgb numpy array"""
    bgr = cv2.imread(filename)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def imwrite_rgb(filename, image, skip_empty=True):
    """Write rgb image to filename"""
    if skip_empty:
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            print("Skipping empty image {0}".format(image))
            return

    if len(image.shape) == 2 or image.shape[2] == 1:
        bgr = cv2.cvtColor(np.uint8(image), cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2BGR)
    return cv2.imwrite(filename, bgr)


def create_rgb_with_size(height, width):
    """Return 3-channel rgb image with given height/width"""
    return np.zeros((height, width, 3), np.uint8)


def add_box_to_image(image, box, rgb_value):
    """Set values in image within box to rgb_value"""
    xmin, xmax, ymin, ymax = box

    print("box {0} rgb_value {1}".format(box, rgb_value))
    image[ymin:ymax, xmin:xmax] = rgb_value
    return image


def resize_with_padding(image, desired_size):
    old_size = image.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format
    image = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return new_image


def crop_to_region_containing_mask(image, image_labels, buffer=5):

    # Coordinates of non-black pixels.
    bw_mask = cv2.cvtColor(image_labels, cv2.COLOR_RGB2GRAY)
    mask = bw_mask != 0
    coords = np.argwhere(mask)

    if len(coords) > 0:
        # Bounding box of non-black pixels.
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top

        x_size, y_size, _ = image.shape

        if x0 > buffer:
            x0 -= buffer
        if y0 > buffer:
            y0 -= buffer
        if x1 + buffer < x_size:
            x1 += buffer
        if y1 + buffer < y_size:
            y1 += buffer

        # print("cropped region: ", x0, x1, y0, y1)

        # Get the contents of the bounding box.
        cropped_image = image[x0:x1, y0:y1]
        cropped_mask = image_labels[x0:x1, y0:y1]
        return cropped_image, cropped_mask
    return None, None


def expand_slice(slice_obj, x_offset, y_offset, bounds):
    start_0 = slice_obj[0].start - x_offset
    stop_0 = slice_obj[0].stop + x_offset
    start_1 = slice_obj[1].start - y_offset
    stop_1 = slice_obj[1].stop + y_offset

    xmin, xmax, ymin, ymax = bounds

    if start_0 < xmin:
        start_0 = xmin
    if stop_0 > xmax:
        stop_0 = xmax
    if start_1 < ymin:
        start_t = ymin
    if stop_1 > ymax:
        start_1 = ymax

    return np.s_[start_0:stop_0, start_1:stop_1]


def region_size(roi):
    if roi is None:
        return 0
    return (roi[0].start + roi[0].stop) * (roi[1].start + roi[1].stop)


def select_segments(image, image_labels, mask_classes, buffer=10):

    copy = np.zeros(image.shape[:2], np.uint8)

    for i, class_value in enumerate(mask_classes):
        print i, class_value
        equality = np.equal(image_labels, class_value)
        class_map = np.all(equality, axis=-1)
        copy[class_map] = i

    regions = find_objects(copy)

    print(regions)

    regions = [i for i in regions if i != None]

    if len(regions) == 0:
        return []

    image_shape = image.shape[:2]
    bounds = [0, image_shape[0], 0, image_shape[1]]

    images = []
    for roi in regions:
        roi_with_buffer = expand_slice(roi,
                                       buffer,
                                       buffer,
                                       bounds
                                       )
        images.append(
            [image[roi_with_buffer], image_labels[roi_with_buffer]]
        )
    return images


def select_biggest_segment(image, image_labels, mask_classes, buffer=10):

    copy = np.zeros(image.shape[:2], np.uint8)

    for i, class_value in enumerate(mask_classes):
        equality = np.equal(image_labels, class_value)
        class_map = np.all(equality, axis=-1)
        copy[class_map] = i

    regions = find_objects(copy)

    print(regions)

    if len(regions) == 0:
        return None, None

    region_sizes = [region_size(i) for i in regions]
    biggest_region = regions[np.argmax(region_sizes)]

    image_shape = image.shape[:2]

    bounds = [0, image_shape[0], 0, image_shape[1]]

    region_with_buffer = expand_slice(biggest_region,
                                      buffer,
                                      buffer,
                                      bounds
                                      )

    print("region with buffer", region_with_buffer)
    return image[region_with_buffer], image_labels[region_with_buffer]


def gen_batch_function(dataset_parameters,
                       region_selection=None  # "region_containing_labels" "region_containing_biggest_segment"
                       ):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """

    training_images = dataset_parameters.training_images()

    training_image_paths, validation_image_paths = train_test_split(training_images, test_size=0.05)

    def make_get_batches_fn(image_paths):
        def get_batches_fn(batch_size):
            """
            Create batches of training data
            :param batch_size: Batch Size
            :return: Batches of training data
            """
            random.shuffle(image_paths)
            for batch_i in range(0, len(image_paths), batch_size):
                images = []
                gt_images = []
                for image_file in image_paths[batch_i:batch_i+batch_size]:
                    gt_image_file = dataset_parameters.groundtruth_image_from_source_image(image_file)

                    raw_image = scipy.misc.imread(image_file)
                    raw_groundtruth_image = scipy.misc.imread(gt_image_file)

                    if region_selection == "region_containing_labels":
                        image, groundtruth_image = crop_to_region_containing_mask(raw_image, raw_groundtruth_image)
                    elif region_selection == "region_containing_biggest_segment":
                        image, groundtruth_image = select_biggest_segment(raw_image, raw_groundtruth_image, dataset_parameters.MASK_CLASSES)
                    else:
                        image, groundtruth_image = raw_image, raw_groundtruth_image

                    if image is None or groundtruth_image is None:
                        print("Skipping image {0}".format(image_file))
                        continue

                    image = resize_with_padding(image, dataset_parameters.image_shape[0])
                    groundtruth_image = resize_with_padding(groundtruth_image, dataset_parameters.image_shape[0])

                    # cv2.imwrite(os.path.basename(image_file), image)
                    # image = scipy.misc.imresize(
                    #     cropped_image,
                    #     dataset_parameters.image_shape
                    # )
                    # groundtruth_image = scipy.misc.imresize(
                    #     cropped_groundtruth_image,
                    #     dataset_parameters.image_shape)

                    onehot_groundtruth = one_hot_it(groundtruth_image,
                                                    dataset_parameters.MASK_CLASSES)

                    images.append(image)
                    gt_images.append(onehot_groundtruth)
                if len(images) > 0:
                    yield np.array(images), np.array(gt_images)
                else:
                    continue

        return get_batches_fn

    training_generator = make_get_batches_fn(training_image_paths)
    validation_generator = make_get_batches_fn(validation_image_paths)

    training_generator.num_samples = len(training_image_paths)
    validation_generator.num_samples = len(validation_image_paths)
    return training_generator, validation_generator


def gen_test_output(test_images, dataset_parameters, sess, logits, keep_prob, image_placeholder):
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

    for image_file in test_images:
        raw_image = scipy.misc.imread(image_file)
        image = resize_with_padding(raw_image, dataset_parameters.image_shape[0])
        # image = scipy.misc.imresize(raw_image, dataset_parameters.image_shape)

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
        # yield image_file, np.array(labeled_image)


def save_inference_samples(test_images,
                           dataset_parameters,
                           sess,
                           logits,
                           keep_prob,
                           input_image,
                           directory_suffix=None
                           ):

    output_dir = dataset_parameters.output_dir()

    if directory_suffix is not None:
        output_dir = os.path.join(output_dir, "{0:03}".format(directory_suffix))

    # Make folder for output
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    model_outputs = gen_test_output(
        test_images,
        dataset_parameters,
        sess,
        logits,
        keep_prob,
        input_image
    )

    for i, (path, image) in enumerate(model_outputs):
        target_file = os.path.join(output_dir, "{0:05}_{1}".format(i, os.path.basename(path)))
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


def recursive_glob(directory, *patterns):
    matches = []
    for root, _, filenames in os.walk(directory):
        for pattern in patterns:
            for filename in fnmatch.filter(filenames, pattern):
                matches.append(os.path.join(root, filename))
    return matches


class DatasetParameters:
    model_name = None

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def model_dir(self):
        return os.path.join(self.data_dir, self.model_name)

    def model_savedir(self):
        return os.path.join(self.model_dir(), "model")

    def model_savepath(self):
        return os.path.join(self.model_savedir(), "saved_model")

    def output_dir(self):
        return os.path.join(self.model_dir(), "predictions")

    def validation_preview_images(self):
        return recursive_glob(os.path.join(self.data_dir, self.model_name + "-preview"), '*.png', '*.jpg')


def filter_images_without_groundtruth(images):
    filtered = []
    for image in images:
        directory, base = os.path.dirname(image), os.path.basename(image)
        groundtruth_image = os.path.join(directory, "groundtruth_" + base)
        if os.path.exists(groundtruth_image):
            filtered.append(image)
    return filtered


class TrafficLightParameters(DatasetParameters):
    """
    Use this to train a semantic segmentation network using data from the bosch traffic light dataset: https://hci.iwr.uni-heidelberg.de/node/6132
    The network will predict for each pixel whether that pixel belongs to a light in red-light/green-light/yellow-light state or is an 'unknown'
    """
    model_name = "trafficlight-segmenter"
    num_classes = 4
    image_shape = (256, 256)

    MASK_CLASSES = (
        (0, 0, 0),
        (255, 0, 0),  # red
        (0, 255, 0),  # green
        (255, 255, 0)  # yellow
    )

    def training_images(self):
        dataset_dir = os.path.join(self.data_dir, "bosch_train")
        all_image_paths = recursive_glob(dataset_dir, '*.png', "*.jpg")
        filtered = [i for i in all_image_paths if not 'groundtruth' in i]

        # make sure that groundtruth images exists for each image in filtered
        filtered = filter_images_without_groundtruth(filtered)

        return filtered

    def test_images(self):
        return self.training_images()

    def groundtruth_image_from_source_image(self, image_path):
        return os.path.join(os.path.dirname(image_path), "groundtruth_" + os.path.basename(image_path))


class RoadParameters(DatasetParameters):
    """Road segmentation dataset"""
    model_name = "road-segmenter"
    num_classes = 3
    image_shape = (256, 320)

    MASK_CLASSES = (
        (0, 0, 0),
        (255, 0, 0),
        (255, 0, 255)
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()

    parser.add_argument("--bosch-file", help="Path to bosch yaml file", required=False)
    parser.add_argument("--labelme-dir", help="Path to directory containing exports from labelme", required=False)
    parser.add_argument("--lisa-dir", help="Path to directory containing lisa traffic dataset", required=False)

    args = parser.parse_args()

    if args.labelme_dir:
        data, source_dir = load_labelme(args.labelme_dir)
        import_bosch_style_data(data, source_dir)

    if args.bosch_file:
        data, source_dir = load_bosch(args.bosch_file)
        import_bosch_style_data(data, source_dir)

    if args.lisa_dir:
        csv_files = recursive_glob(args.lisa_dir, "*BOX.csv")
        print("Processing files: ", csv_files)
        for csv_file in csv_files:
            data, source_dir = load_lisa(csv_file)
            import_bosch_style_data(data, source_dir)


def load_bosch(yml_file):
    bosch_data = []
    source_file = yml_file
    source_dir = os.path.dirname(yml_file)

    with open(source_file, 'r') as f:
        bosch_data = yaml.safe_load(f)
        return bosch_data, source_dir


def load_lisa(lisa_csv_file):
    import csv

    data = {}

    with open(lisa_csv_file, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile, delimiter=';')
        for row in csvreader:
            path = row["Filename"].replace("dayTraining", "frames")

            classlabel = row["Annotation tag"]

            if classlabel == "stop" or classlabel == "stopLeft":
                label = "Red"
            elif classlabel == "go" or classlabel == "goForward" or classlabel == "goLeft":
                label = "Green"
            elif classlabel == "warning" or classlabel == "warningLeft":
                label = "Yellow"
            else:
                raise Exception("Unknown label {0}".format(classlabel))

            xmin = row["Upper left corner X"]
            ymin = row["Upper left corner Y"]
            xmax = row["Lower right corner X"]
            ymax = row["Lower right corner Y"]

            box = {}

            box["x_min"] = xmin
            box["x_max"] = xmax
            box["y_min"] = ymin
            box["y_max"] = ymax
            box["label"] = label

            if data.get(path, None) is None:
                item = {}
                item["path"] = path
                item["boxes"] = [box]
                data[path] = item
            else:
                item = data[path]
                item["boxes"].append(box)

    # turn back into list
    items = data.values()

    print("{0} items in file {1}".format(len(items), lisa_csv_file))

    return items, os.path.dirname(lisa_csv_file)


def load_labelme(base_dir):
    import xmltodict
    import json
    xmlfiles = recursive_glob(base_dir, "*.xml")

    data = []

    for xmlfile in xmlfiles:
        with open(xmlfile, 'r') as f:
            labelme = xmltodict.parse(f.read())

            item = labelme['annotation']

            print(json.dumps(item, indent=4))

            image_path = os.path.join("Images", item['folder'], item['filename'])

            # hack -- changed collection name because collection's can only contain 20 images ...
            image_path = image_path.replace("parking_lot_images", "greenlight")

            box = {}

            xmin = 9999999
            xmax = 0
            ymin = 9999999
            ymax = 0

            for point in item['object']['polygon']['pt']:
                x, y = int(point['x']), int(point['y'])
                print("x,y", x, y)
                if x < xmin:
                    xmin = x
                if x > xmax:
                    xmax = x
                if y < ymin:
                    ymin = y
                if y > ymax:
                    ymax = y

            classlabel = item['object']['name']

            if classlabel == "redlight":
                label = "Red"
            elif classlabel == "greenlight":
                label = "Green"
            elif classlabel == "yellowlight":
                label = "Yellow"
            else:
                raise Exception("Error unknown label")

            box["x_min"] = xmin
            box["x_max"] = xmax
            box["y_min"] = ymin
            box["y_max"] = ymax
            box["label"] = label

            newitem = {}
            newitem['path'] = image_path
            # not generally correct -- but there is only one annotation per
            # image in this dataset so this will work...
            newitem['boxes'] = [box]

            print(json.dumps(newitem, indent=4))

            data.append(newitem)
    return data, base_dir


def import_bosch_style_data(bosch_data, source_dir):

    for item in bosch_data:
        source_path = os.path.join(source_dir, item.get("path"))

        print("Reading image info from path {0}".format(source_path))
        image = imread_rgb(source_path)
        print("Image dimensions: {0}".format(image.shape))
        size = image.shape[:2]
        groundtruth_image = create_rgb_with_size(size[0], size[1])
        for box in item.get("boxes", []):
            # if box.get('occluded', True):
            #     print("Skipping occluded box")
            #     continue
            label = box.get('label', None)
            if label == "Red":
                mask_value = TrafficLightParameters.MASK_CLASSES[1]
            elif label == "Green":
                mask_value = TrafficLightParameters.MASK_CLASSES[2]
            elif label == "Yellow":
                mask_value = TrafficLightParameters.MASK_CLASSES[3]
            elif label == "RedLeft":
                # not ideal ...
                mask_value = TrafficLightParameters.MASK_CLASSES[1]
            elif label == "GreenLeft":
                # not ideal ...
                mask_value = TrafficLightParameters.MASK_CLASSES[2]
            else:
                # we can't do anything with other guesses ...
                mask_value = TrafficLightParameters.MASK_CLASSES[0]
            xmin = int(box.get('x_min'))
            xmax = int(box.get('x_max'))
            ymin = int(box.get('y_min'))
            ymax = int(box.get('y_max'))
            groundtruth_image = add_box_to_image(groundtruth_image, (xmin, xmax, ymin, ymax), mask_value)

        groundtruth_image_path = os.path.join(
            os.path.dirname(source_path),
            "groundtruth_" + os.path.basename(source_path)
        )

        print("Writing groundtruth image to {0}".format(os.path.abspath(groundtruth_image_path)))

        imwrite_rgb(groundtruth_image_path, groundtruth_image)

        segments = select_segments(image, groundtruth_image, TrafficLightParameters.MASK_CLASSES)

        for i, (image_s, groundtruth_s) in enumerate(segments):
            image_s_path = os.path.join(
                os.path.dirname(source_path),
                "{0:02}_".format(i) + os.path.basename(source_path)
            )
            groundtruth_s_path = os.path.join(
                os.path.dirname(source_path),
                "groundtruth_" + "{0:02}_".format(i) + os.path.basename(source_path)
            )

            print("Writing image segment to {0}".format(os.path.abspath(image_s_path)))
            imwrite_rgb(image_s_path, image_s)

            print("Writing groundtruth image segment to {0}".format(os.path.abspath(groundtruth_s_path)))
            imwrite_rgb(groundtruth_s_path, groundtruth_s)


if __name__ == "__main__":
    main()
