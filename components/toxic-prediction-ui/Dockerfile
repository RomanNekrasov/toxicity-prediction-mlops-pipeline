FROM python:3.10.7-slim
WORKDIR /pipelines/component/src
ADD src/ ./
ADD requirements.txt ./
EXPOSE 5002
RUN pip install -r requirements.txt
CMD ["python3","src/app.py"]