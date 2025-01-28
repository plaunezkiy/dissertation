FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /model

COPY requirements.txt .

RUN pip install -r requirements.txt

ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models

copy . .
