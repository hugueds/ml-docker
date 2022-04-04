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


model_name = 'cats-dogs'

def create_model(input_shape, num_classes):
    """
    Create a MobileNet model with the given input shape and number of classes.
    """
    base_model = MobileNet(input_shape=input_shape, include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model


def create_generators(train_dir, validation_dir, batch_size):
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
        horizontal_flip=True,
        fill_mode='nearest')

    validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

    return train_generator, validation_generator

train_dir = f'data/{model_name}/train'
validation_dir = f'data/{model_name}/validation'
batch_size = 8 

train_generator, validation_generator = create_generators(train_dir, validation_dir, batch_size)

# create the model

# read the number of classes from the generator
num_classes = len(train_generator.class_indices)

model = create_model(input_shape=(224, 224, 3), num_classes=num_classes)

# compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(
    train_generator,
    steps_per_epoch=1,
    epochs=1,
    validation_data=validation_generator,
    validation_steps=1,
    callbacks=[
        ModelCheckpoint(f'./models/{model_name}/mobilenet_model_ckpt.h5', monitor='val_acc', save_best_only=True),
        EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='auto'),
        ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, mode='auto', factor=0.1, min_lr=0.00001)
    ])


# save the model
model.save('./models/mobilenet_model.h5')

# plot the model

# evaluate the model
scores = model.evaluate_generator(validation_generator, steps=50)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


# predict the class
# img_path = f'./data/{model_name}/test/1.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
# preds = model.predict(x)
# print('Predicted:', decode_predictions(preds))








