from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D, Input
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model
from keras import backend as K
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint

CLASS_LABELS = [
    "unknown",
    "red",
    "yellow",
    "green"
]


def make_simple():
    num_classes = 4

    model = Sequential()
    model.add(Lambda(lambda x: (x - 128) / 128, input_shape=(150, 150, 3)))
    # model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu', padding='valid'))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu', padding='valid'))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='valid'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1164))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


modelname_template = "tl_classifier" + "_{epoch:02d}.hdf5"


DEFAULT_TRAINING_DIRECTORY = "../../../../../training_images/"
DEFAULT_VALIDATION_DIRECTORY = "../../../../../validation_images/"


def main():

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', help='Path to directory containing labeled training images', default=DEFAULT_TRAINING_DIRECTORY)
    parser.add_argument('--validation_dir', help='Path to directory containing labeled validation images', default=DEFAULT_VALIDATION_DIRECTORY)

    args = parser.parse_args()

    training_directory = args.train_dir
    validation_directory = args.validation_dir

    train_datagen = ImageDataGenerator(
        rescale=None,  # 1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    validation_datagen = ImageDataGenerator(
        rescale=None  # 1./255
    )

    train_generator = train_datagen.flow_from_directory(
        training_directory,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        classes=CLASS_LABELS
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_directory,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        classes=CLASS_LABELS
    )

    model = make_simple()

    # balance classes ...?
    # counter = Counter(train_generator.classes)
    # max_val = float(max(counter.values()))
    # class_weights = {class_id: max_val/num_images for class_id, num_images in counter.items()}

    model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=10,
        callbacks=[ModelCheckpoint(filepath=modelname_template)]
    )


if __name__ == "__main__":
    main()
