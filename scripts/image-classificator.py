# create a tensorflow keras model for mobilenet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
# import mobilenet v2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import Callback
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
import os
# load the load_img and img_to_array functions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


# receive model name as argument
parser = ArgumentParser()
parser.add_argument("--model", help="model name")
args = parser.parse_args()

# if model name is not given, exit and print error message
if not parser.parse_args().model:
    print("Please specify a model name")
    exit()

model_name = args.model

def create_model(base_model, input_shape, num_classes):

    if base_model == 'mobilenet':
        model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    elif base_model == 'resnet':
        model = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')

    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=model.input, outputs=x)

# load images as np arrays
def load_images():

    # open the folder with the images
    train_dir = '../data/{}/dataset'.format(model_name)

    # get the number of classes
    num_classes = len(os.listdir(train_dir))

    # get the input shape
    input_shape = (224, 224, 3)

    # open the images as np arrays and append the labels accordingly to its folder
    X = []
    y = []

    for i, class_name in enumerate(os.listdir(train_dir)):
        for img in os.listdir(os.path.join(train_dir, class_name)):
            img_path = os.path.join(train_dir, class_name, img)
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img)
            X.append(img_array)
            y.append(i)

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # convert to categorical
    y = to_categorical(y, num_classes)

    # split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val, num_classes


def create_generators(X_train, X_val, y_train, y_val, batch_size):
    """
    Create training and validation image generators for the model using the
    given directory locations, a batch size for both, and the input shape.
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest')

    validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # train generator based on np arrays
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)

    # validation generator based on np arrays
    validation_generator = validation_datagen.flow(X_val, y_val, batch_size=batch_size)

    return train_generator, validation_generator

def train(train_generator, validation_generator, num_classes, epochs, batch_size):

    # create the model
    model = create_model('resnet', input_shape=(224, 224, 3), num_classes=num_classes)

    # compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model
    model.fit(
        train_generator,
        steps_per_epoch=40,
        epochs=4,
        validation_data=validation_generator,
        validation_steps=4,
        callbacks=[
            ModelCheckpoint(f'../models/{model_name}/mobilenet_model_ckpt.h5', monitor='accuracy', save_best_only=True),
            EarlyStopping(monitor='accuracy', patience=5, verbose=1, mode='auto'),
            ReduceLROnPlateau(monitor='accuracy', patience=3, verbose=1, mode='auto', factor=0.1, min_lr=0.00001)
        ])

    # save the model
    model.save('../models/{}.h5'.format(model_name))

    # evaluate the model
    scores = Model.evaluate(model, validation_generator, steps=5)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


def predict():
    # predict the class

    # get all the images in the test directory
    test_dir = '../data/{}/test'.format(model_name)

    # create an arry of all the images
    X_test = []

    # load the class names
    class_names = os.listdir('../data/{}/dataset/'.format(model_name))

    # load the images as np arrays
    for img in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img)
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        X_test.append(img_array)


    # convert to numpy arrays
    X_test = np.array(X_test)

    # load the model
    model = load_model('../models/{}.h5'.format(model_name))

    # predict the class
    predictions = model.predict(X_test)

    # print the predictions with the class names
    for i, prediction in enumerate(predictions):

        # get the index of the highest probability
        index = np.argmax(prediction)

        # get the class name
        class_name = class_names[index]

        # print the prediction
        print('{} - {}'.format(img, class_name))


if __name__ == '__main__':
    # load the images
    X_train, X_val, y_train, y_val, num_classes = load_images()

    # create the generators
    train_generator, validation_generator = create_generators(X_train, X_val, y_train, y_val, batch_size=32)

    # train the model
    train(train_generator, validation_generator, num_classes, epochs=100, batch_size=8)

    # predict()
