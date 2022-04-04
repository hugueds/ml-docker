FROM python:3.9

ENV TFINPUT_DIR=$TF_INPUT_DIR
ENV TF_OUTPUT_DIR=$TF_OUTPUT_DIR
ENV TF_TENSOR_SIZE=$TF_TENSOR_SIZE
ENV TF_EPOCHS=$TF_EPOCHS
ENV TF_BATCH_SIZE=$TF_BATCH_SIZE
ENV TF_LEARNING_RATE=$TF_LEARNING_RATE
ENV TF_DROPOUT=$TF_DROPOUT

WORKDIR /app/scripts

RUN mkdir -p /app/notebooks
RUN mkdir -p /app/data
RUN mkdir -p /app/scripts
RUN mkdir -p /app/models

# RUN git clone https://github.com/tensorflow/models.git /app/models

COPY requirements.txt /app/requirements.txt

RUN apt update && apt upgrade -y
RUN apt install protobuf-compiler python3-pip -y

RUN pip install --upgrade pip
RUN pip install sklearn
RUN pip install tensorflow
RUN pip install mlflow
RUN pip install -r /app/requirements.txt

# convert the notebook to a python script
# convert the tensoflow model to a tf.js model


# ENTRYPOINT ["python", "index.py"]

