from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D, Input, BatchNormalization, GlobalAveragePooling2D, Concatenate, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model
from keras import backend as K
from keras import optimizers, regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

import os
import cv2
import numpy as np

from collections import Counter


CLASS_LABELS = [
    "unknown",
    "red",
    "yellow",
    "green"
]

COLOR_IMAGE_SIZE = (250, 250, 3)
BW_IMAGE_SIZE = (250, 250, 1)


def make_squeeze_net(input_shape, nb_classes=4):
    """ Keras Implementation of SqueezeNet(arXiv 1602.07360)

    Arguments:
    input_shape -- shape of the input images (rows, cols, channels)
    nb_classes  -- total number of final categories

    Based on implementation from here: https://github.com/DT42/squeezenet_demo/blob/master/model.py
    From paper: https://arxiv.org/pdf/1602.07360.pdf
    """

    input_img = Input(shape=input_shape)
    conv1 = Conv2D(
        96, (7, 7), activation='relu', kernel_initializer='glorot_uniform',
        strides=(2, 2), padding='same', name='conv1',
        data_format="channels_last")(input_img)
    maxpool1 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool1',
        data_format="channels_last")(conv1)
    fire2_squeeze = Conv2D(
        16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_squeeze',
        data_format="channels_last")(maxpool1)
    fire2_expand1 = Conv2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand1',
        data_format="channels_last")(fire2_squeeze)
    fire2_expand2 = Conv2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand2',
        data_format="channels_last")(fire2_squeeze)
    merge2 = Concatenate(axis=3)([fire2_expand1, fire2_expand2])

    fire3_squeeze = Conv2D(
        16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_squeeze',
        data_format="channels_last")(merge2)
    fire3_expand1 = Conv2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_expand1',
        data_format="channels_last")(fire3_squeeze)
    fire3_expand2 = Conv2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_expand2',
        data_format="channels_last")(fire3_squeeze)
    merge3 = Concatenate(axis=3)([fire3_expand1, fire3_expand2])

    fire4_squeeze = Conv2D(
        32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_squeeze',
        data_format="channels_last")(merge3)
    fire4_expand1 = Conv2D(
        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand1',
        data_format="channels_last")(fire4_squeeze)
    fire4_expand2 = Conv2D(
        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand2',
        data_format="channels_last")(fire4_squeeze)
    merge4 = Concatenate(axis=3)([fire4_expand1, fire4_expand2])
    maxpool4 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool4',
        data_format="channels_last")(merge4)

    fire5_squeeze = Conv2D(
        32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_squeeze',
        data_format="channels_last")(maxpool4)
    fire5_expand1 = Conv2D(
        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_expand1',
        data_format="channels_last")(fire5_squeeze)
    fire5_expand2 = Conv2D(
        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_expand2',
        data_format="channels_last")(fire5_squeeze)
    merge5 = Concatenate(axis=3)([fire5_expand1, fire5_expand2])

    fire6_squeeze = Conv2D(
        48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_squeeze',
        data_format="channels_last")(merge5)
    fire6_expand1 = Conv2D(
        192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_expand1',
        data_format="channels_last")(fire6_squeeze)
    fire6_expand2 = Conv2D(
        192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_expand2',
        data_format="channels_last")(fire6_squeeze)
    merge6 = Concatenate(axis=3)([fire6_expand1, fire6_expand2])

    fire7_squeeze = Conv2D(
        48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_squeeze',
        data_format="channels_last")(merge6)
    fire7_expand1 = Conv2D(
        192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_expand1',
        data_format="channels_last")(fire7_squeeze)
    fire7_expand2 = Conv2D(
        192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_expand2',
        data_format="channels_last")(fire7_squeeze)
    merge7 = Concatenate(axis=3)([fire7_expand1, fire7_expand2])

    fire8_squeeze = Conv2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_squeeze',
        data_format="channels_last")(merge7)
    fire8_expand1 = Conv2D(
        256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_expand1',
        data_format="channels_last")(fire8_squeeze)
    fire8_expand2 = Conv2D(
        256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_expand2',
        data_format="channels_last")(fire8_squeeze)
    merge8 = Concatenate(axis=3)([fire8_expand1, fire8_expand2])

    maxpool8 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool8',
        data_format="channels_last")(merge8)
    fire9_squeeze = Conv2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire9_squeeze',
        data_format="channels_last")(maxpool8)
    fire9_expand1 = Conv2D(
        256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire9_expand1',
        data_format="channels_last")(fire9_squeeze)
    fire9_expand2 = Conv2D(
        256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire9_expand2',
        data_format="channels_last")(fire9_squeeze)
    merge9 = Concatenate(axis=3)([fire9_expand1, fire9_expand2])

    fire9_dropout = Dropout(0.5, name='fire9_dropout')(merge9)
    conv10 = Conv2D(
        nb_classes, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='valid', name='conv10',
        data_format="channels_last")(fire9_dropout)

    global_avgpool10 = GlobalAveragePooling2D(data_format='channels_last')(conv10)
    softmax = Activation("softmax", name='softmax')(global_avgpool10)

    model = Model(inputs=input_img, outputs=softmax)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    return model


def make_simple(input_shape, num_classes=4):
    model = Sequential()
    # model.add(Lambda(lambda x: (x - 128) / 128, input_shape=image_size))
    # model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Conv2D(12, kernel_size=(5, 5), strides=(2, 2), activation='relu', padding='valid', input_shape=input_shape))
    model.add(Conv2D(12, kernel_size=(5, 5), strides=(2, 2), activation='relu', padding='valid'))
    model.add(Conv2D(12, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='valid'))
    # model.add(Conv2D(12, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='valid'))
    # model.add(Conv2D(12, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='valid'))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    return model


modelname_template = "tl_classifier" + "_{epoch:02d}.hdf5"


def main():

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", help="Path to directory containing labeled training images", default="../../../../../vmshared")
    parser.add_argument("--mode", help="Whether to train or categorize images", default="train", choices=["train", "test"])
    parser.add_argument("--color-mode", help="Feed grayscale or multi-channel image into classifer model?", default="grayscale")
    parser.add_argument("--save-generated-images", help="Save the generated images for human verification", default=False, action='store_true')
    parser.add_argument("--use-bosch", help="Train from bosch images", default=False, action="store_true")
    parser.add_argument("--batch-size", help="Minibatch size", default=32, type=int)
    parser.add_argument("--model", help="Select model to use", choices=["simple", "squeezenet"], default="simple")

    args = parser.parse_args()

    training_directory = "{0}/training_images".format(args.data_dir)
    validation_directory = "{0}/validation_images".format(args.data_dir)

    if args.use_bosch:
        training_directory = "{0}/bosch_labeled".format(args.data_dir)

    if args.color_mode == "grayscale":
        input_shape = BW_IMAGE_SIZE
        color_mode = "grayscale"
    else:
        input_shape = COLOR_IMAGE_SIZE
        color_mode = "rgb"

    print("Color mode: {0} -- input shape {1}".format(color_mode, input_shape))

    training_generated_images = "{0}/augmented_training_samples".format(args.data_dir) if args.save_generated_images else None
    validation_generated_images = "{0}/augmented_validation_samples".format(args.data_dir) if args.save_generated_images else None
    batch_size = args.batch_size
    model_choice = args.model

    if args.mode == "train":

        if training_generated_images and not os.path.exists(training_generated_images):
            os.makedirs(training_generated_images)
        if validation_generated_images and not os.path.exists(validation_generated_images):
            os.makedirs(validation_generated_images)

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=1.5,
            height_shift_range=0.2,
            width_shift_range=0.3,
            channel_shift_range=0.1,
            data_format="channels_last"
        )

        # train_datagen = ImageDataGenerator(data_format="channels_last")

        # validation_datagen = ImageDataGenerator(
        #     rescale=1./255,
        #     shear_range=0.2,
        #     zoom_range=0.2,
        #     height_shift_range=0.2,
        #     width_shift_range=0.3,
        #     channel_shift_range=0.1,
        #     data_format="channels_last"
        # )

        validation_datagen = ImageDataGenerator(
            rescale=1./255,
            data_format="channels_last")

        train_generator = train_datagen.flow_from_directory(
            training_directory,
            color_mode=color_mode,
            target_size=input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            classes=CLASS_LABELS,
            shuffle=True,
            save_to_dir=training_generated_images,
        )

        validation_generator = validation_datagen.flow_from_directory(
            validation_directory,
            color_mode=color_mode,
            target_size=input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            classes=CLASS_LABELS,
            save_to_dir=validation_generated_images
        )

        if model_choice == "simple":
            model = make_simple(input_shape)
        elif model_choice == "squeezenet":
            model = make_squeeze_net(input_shape)

        train_counter = Counter(train_generator.classes)
        train_steps_per_epoch = sum(train_counter.values()) // batch_size

        validation_counter = Counter(validation_generator.classes)
        validation_steps_per_epoch = sum(validation_counter.values()) // batch_size

        print("Training steps: ", train_steps_per_epoch)
        print("Validation steps: ", validation_steps_per_epoch)

        # balance unbalanced classes ...?
        max_val = float(max(train_counter.values()))
        class_weight = {class_id: max_val/num_images for class_id, num_images in train_counter.items()}
        print("class_weights", class_weight)

        model.fit_generator(
            train_generator,
            steps_per_epoch=train_steps_per_epoch,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=validation_steps_per_epoch,
            callbacks=[ModelCheckpoint(filepath=modelname_template, monitor="val_loss", save_best_only=True)],
        )
    else:
        print("Testing model")
        model_name = "tl_classifier_09.hdf5"
        model = load_model(model_name)

        input_shape = tuple([i.value for i in model.input.get_shape()[1:]])

        print("Input shape from model {0}".format(input_shape))

        if input_shape[2] == 1:
            # grayscale
            def load_image(image_path):
                grayscale = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                resized = cv2.resize(grayscale, input_shape[:2])
                resized = np.expand_dims(image, axis=len(image.shape))
                return resized
        else:
            # rgb
            def load_image(image_path):
                rgb_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                return cv2.resize(rgb_image, input_shape[:2])

        for directory in [validation_directory]:
            print(directory)
            for category in os.listdir(directory):
                for filename in os.listdir(os.path.join(directory, category)):
                    if filename.endswith(".jpg"):
                        path_to_file = os.path.join(directory, category, filename)
                        image = load_image(path_to_file)
                        prediction = model.predict_classes(np.expand_dims(image, axis=0), verbose=False)
                        print("Prediction for image {0} => {1}".format(path_to_file, CLASS_LABELS[prediction[0]]))


if __name__ == "__main__":
    main()
