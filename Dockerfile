FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip git && \
    rm -rf /var/lib/apt/lists/*


COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


COPY . .


# The container can be run with commands like:
# docker run --gpus all -it <image_name> python main.py ...
# docker run --gpus all -it <image_name> python inference.py ...
