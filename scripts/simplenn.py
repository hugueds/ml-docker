# crete a basic neural network with keras and tensorflow
import tensorflow as tf
import numpy as np
import mlflow

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=[1])
    ])

    model.compile(optimizer='sgd', loss='mean_squared_error')
    return model

def train_model(model, inputs, outputs, epochs):
    model.fit(inputs, outputs, epochs=epochs)

def predict_model(model, inputs):
    return model.predict(inputs).flatten()

def main():

    inputs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=float)
    outputs = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0], dtype=float)

    model = create_model()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("tensorflow")

    mlflow.start_run()

    train_model(model, inputs, outputs, epochs=1000)
    print(predict_model(model, np.array([10.0])))
    # evaluate the model

    mlflow.end_run()

    # save the model
    mlflow.keras.log_model(model, "model")

    mlflow.log_artifact("index.py")




if __name__ == '__main__':
    main()
















