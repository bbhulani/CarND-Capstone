import yaml
import os
import shutil

import cv2
import numpy as np


MASK_COLORS = {
    "unknown": (0, 0, 0),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "yellow": (255, 255, 0),
}


def imread_rgb(filename):
    """Read image file from filename and return rgb numpy array"""
    bgr = cv2.imread(filename)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def imwrite_rgb(filename, image):
    """Write rgb image to filename"""
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()

    parser.add_argument("--bosch-file", help="Path to bosch yaml file", required=True)

    args = parser.parse_args()

    source_file = args.bosch_file
    source_dir = os.path.dirname(source_file)

    with open(source_file, 'r') as f:
        data = yaml.safe_load(f)

        size = None
        for item in data:
            source_path = os.path.join(source_dir, item.get("path"))

            if size is None:
                print("Reading image info from path {0}".format(source_path))
                img = imread_rgb(source_path)
                size = img.shape[:2]
                print("Image dimensions: {0}".format(img.shape))
            image = create_rgb_with_size(size[0], size[1])
            for box in item.get("boxes", []):
                # if box.get('occluded', True):
                #     print("Skipping occluded box")
                #     continue
                label = box.get('label', None)
                if label == "Red":
                    mask_value = MASK_COLORS["red"]
                elif label == "Green":
                    mask_value = MASK_COLORS["green"]
                elif label == "Yellow":
                    mask_value = MASK_COLORS["yellow"]
                else:
                    mask_value = MASK_COLORS["unknown"]
                xmin = int(box.get('x_min'))
                xmax = int(box.get('x_max'))
                ymin = int(box.get('y_min'))
                ymax = int(box.get('y_max'))
                image = add_box_to_image(image, (xmin, xmax, ymin, ymax), mask_value)

            target_label_path = os.path.join(os.path.dirname(source_path), "groundtruth_" + os.path.basename(source_path))
            print("Writing mask to {0}".format(os.path.abspath(target_label_path)))
            imwrite_rgb(target_label_path, image)


if __name__ == "__main__":
    main()
