This directory contains an implementation of image based traffic light classification.

# tl_classifier.py

Implements communication between ros and traffic light models.

Sends raw images from ros to model(s) and produces a prediction of the
traffic light state represented by the image.

# Keras implementation of classifier to predict traffic light state from whole image

train_classifier.py: keras implementation of the training process for an image classification model with dataset sourced using ImageDataGenerator and flowFromDirectory.

Was originally hoping to implement mvp via this method, but this didn't really seem to get there so started looking into more advanced models. (see below)

# Tensorflow implementation of Semantic Segmentation network to predict traffic light state

## Train the network

0.  Download the bosch dataset zip files
1.  Concatenate the parts and unzip the bosch dataset into a directory called 'bosch_train': `cat dataset_train_rgb.zip.* > dataset_train_rgb.zip; unzip dataset_train_rgb.zip -d ./data_dir/bosch_train`
1.  Import training data from bosch dataset (produces segmented images): `python dataset.py --bosch-file ./path_to/data_dir/bosch_train/train.yaml`
1.  Run the training script providing path to directory containing bosch_train (trained model will be written to this directory): `python train_segmenter.py --mode train --data-dir ./path_to/data_dir --dataset traffic-lights --num-epochs 20 --num-validation-steps 10 --batch-size 10`
1.  Test images will be written each epoch to `./path_to/data_dir/trafficlight-segmenter/predictions`

## Produce some images annotated with the network

0.  `python train_segmenter.py --mode test --dataset traffic-lights --data-dir ./path_to/data_dir`

## Description

train-segmenter.py produces a semantic segmentation network that attempts to predict for each pixel in an image, whether that pixel is part of a 'red light' 'green light' 'yellow light' or none of those.

The network consists of a pre-trained encoding layer based on a fully convolutional version of vgg16 -- then a decoding layer consisting of a stack of convolutional upsampling layers + skip layers to help capture information at different spatial scales.

There was trouble training this network for this task. Because of the unbalanced classes -- the vast majority of the images in the training set consist primarily of 'unknown'/background class. The objects in the training set are fairly small. A couple of strategies were employed to try to work around this:

1.  loss_function="cross_entropy": First attempted to train the network with cross entropy loss function. This led to a network which converged on predicting all background output images due to the extreme class imbalance
2.  loss_function="weighted_cross_entropy": Next tried to apply weighting scheme to classes to see if could correct for the class imbalance -- this also didn't work.
3.  loss_function="iou_estimate": Uses a differentiable estimate of the intersection over union metric -- as loss function. Still wouldn't converge.
4.  loss_function="weighted_iou_estimate": Uses the same differentiable estimate of iou as loss function, but also applies weighting to address unbalanced classes.
5.  Produce cropped training images with higher ratio of non-background to background by selecting the region around the labeled area of interest plus a small buffer of additional pixels
