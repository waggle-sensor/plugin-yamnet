# A Dockerfile is used to define how your code will be packaged. This includes
# your code, the base image and any additional dependencies you need.

# First we choose the base image. For more info, see:
# https://github.com/waggle-sensor/plugin-base-images
# FROM waggle/plugin-base:1.1.0-ml-cuda11.0-amd64
FROM waggle/plugin-base:1.1.1-base

RUN apt-get update \
  && apt-get install -y \
  pulseaudio \
  && rm -rf /var/lib/apt/lists/*

COPY * /app/
RUN pip3 install numpy --upgrade

RUN pip3 install --no-cache-dir -U -r /app/requirements.txt

ARG SAGE_STORE_URL="https://osn.sagecontinuum.org"
ARG BUCKET_ID_MODEL="cafb2b6a-8e1d-47c0-841f-3cad27737698"

ENV SAGE_STORE_URL=${SAGE_STORE_URL} \
    BUCKET_ID_MODEL=${BUCKET_ID_MODEL}

# RUN sage-cli.py storage files download ${BUCKET_ID_MODEL} lite-model_yamnet_tflite_1.tflite --target /app/lite-model_yamnet_tflite_1.tflite

WORKDIR /app
# python3 main.py  --DURATION_S 10 --TOP_K 3
ENTRYPOINT ["python3" , "/app/main.py"]
