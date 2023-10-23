FROM python:3.10.7-slim
WORKDIR /pipelines/component/src
ADD src/component.py ./
ADD requirements.txt ./
ENV MODEL_REPO=models_de2023_group1

RUN pip install -r requirements.txt