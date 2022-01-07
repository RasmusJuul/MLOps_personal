# Base image
FROM anibali/pytorch:1.10.0-cuda11.3-ubuntu20.04

COPY requirements_docker_gpu.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/

# WORKDIR / app/
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]