FROM python:3.10.7-slim
WORKDIR /pipelines/component/src
ADD src/app.py ./
ADD requirements.txt ./
EXPOSE 5001
RUN pip install -r requirements.txt