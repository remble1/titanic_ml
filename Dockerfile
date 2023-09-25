FROM ubuntu:20.04 AS development

COPY ./requirements.txt /usr/src/app/
COPY ./src /src
COPY ./entrypoint.sh /entrypoint.sh

RUN apt-get update;\
    apt-get install -y python3-pip
RUN pip3 install -r /usr/src/app/requirements.txt

FROM development as release