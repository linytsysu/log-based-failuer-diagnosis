FROM registry.cn-shanghai.aliyuncs.com/tcc-public/python:3

WORKDIR /
ADD Dockerfile run.sh drain3.ini preprocess.py feature.py feature2.py main.py requirements.txt /
ADD data /data
ADD prediction_result /prediction_result
ADD nltk_data /nltk_data
RUN pip install --upgrade pip -i https://mirrors.cloud.tencent.com/pypi/simple
RUN pip --no-cache-dir install  -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple

RUN apt -y update
RUN apt install zip

CMD ["sh", "run.sh"]