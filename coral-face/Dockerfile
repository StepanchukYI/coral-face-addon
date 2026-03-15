ARG BUILD_FROM=ghcr.io/home-assistant/aarch64-base-debian:bullseye
FROM ${BUILD_FROM}

# Bullseye has Python 3.9 which is compatible with all Coral packages

# Install Python + system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-numpy \
    python3-pil \
    curl \
    gnupg \
    ca-certificates \
    libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Add Coral Edge TPU repository — all packages from same source for compatibility
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    > /etc/apt/sources.list.d/coral-edgetpu.list \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    libedgetpu1-std \
    python3-pycoral \
    python3-tflite-runtime \
    && rm -rf /var/lib/apt/lists/*

# Flask
RUN pip install --no-cache-dir flask==3.1.0

WORKDIR /opt/coral-face

# Download face detection model (Coral Edge TPU)
RUN mkdir -p models && \
    curl -fsSL -o models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite \
    "https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"

# Download MobileFaceNet embedding model (TFLite CPU)
RUN curl -fsSL -L -o models/mobilefacenet.tflite \
    "https://github.com/MCarlomagno/FaceRecognitionAuth/raw/master/assets/mobilefacenet.tflite"

COPY server.py .
COPY run.sh /
RUN chmod +x /run.sh

EXPOSE 5100

CMD ["/run.sh"]
