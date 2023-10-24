FROM python:3.10.7-slim
ENTRYPOINT python3
WORKDIR /pipelines/component/src
ADD src/ ./
ADD requirements.txt ./
EXPOSE 5002
RUN pip install -r requirements.txt
CMD ["app.py"]