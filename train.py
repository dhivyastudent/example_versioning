import os
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras import applications
from tensorflow.keras.callbacks import CSVLogger
from tqdm.keras import TqdmCallback

# paths
path = os.path.abspath(os.path.dirname(__file__))

img_width, img_height = 150, 150

top_model_weights_path = "bottleneck_fc_model.h5"
train_data_dir = os.path.join(path, "data/train")
validation_data_dir = os.path.join(path, "data/validation")

# dataset sizes (adjust if needed)
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 10
batch_size = 16


def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # load VGG16 without top layer
    model = applications.VGG16(include_top=False, weights='imagenet')

    # training data
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )

    bottleneck_features_train = model.predict(
        generator,
        steps=nb_train_samples // batch_size
    )

    np.save(open('bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)

    # validation data
    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )

    bottleneck_features_validation = model.predict(
        generator,
        steps=nb_validation_samples // batch_size
    )

    np.save(open('bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    train_labels = np.array(
        [0] * (nb_train_samples // 2) +
        [1] * (nb_train_samples // 2)
    )

    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.array(
        [0] * (nb_validation_samples // 2) +
        [1] * (nb_validation_samples // 2)
    )

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_data,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(validation_data, validation_labels),
        verbose=1,
        callbacks=[TqdmCallback(), CSVLogger("metrics.csv")]
    )

    model.save_weights(top_model_weights_path)
    model.save("top_model_complete.h5")


if __name__ == "__main__":
    save_bottleneck_features()
    train_top_model()
