FROM ubuntu:22.10
FROM python:3.7.9


WORKDIR /cage

RUN apt-get update && apt-get install tree -y


COPY ./Requirements.txt /cage/Requirements.txt
RUN pip install -r Requirements.txt

COPY . /cage


RUN pip install -e .



CMD python /cage/CybORG/Evaluation/validation.py && /bin/bash

# CMD /bin/bash