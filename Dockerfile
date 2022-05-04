FROM registry.cn-shanghai.aliyuncs.com/tcc-public/python:3

WORKDIR /
ADD Dockerfile run.sh requirements.txt /
ADD code /code
ADD data /data
ADD user_data /user_data
ADD prediction_result /prediction_result
RUN pip install --upgrade pip -i https://mirrors.cloud.tencent.com/pypi/simple
RUN pip --no-cache-dir install -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple

RUN apt -y update
RUN apt install zip

CMD ["sh", "run.sh"]