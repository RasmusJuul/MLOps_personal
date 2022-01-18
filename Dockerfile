# Base image
FROM python:3.7-slim

# install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data.dvc data.dvc
COPY .git/ .git/
COPY .dvc/ .dvc/


RUN pip install -r requirements.txt --no-cache-dir

RUN pip install dvc[gs]
RUN dvc pull

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]

# Install Cloud SDK
# RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-sdk -y
      