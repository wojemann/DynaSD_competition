FROM python:3.10.8

WORKDIR /model

COPY  requirements.txt /model/
RUN pip install --no-cache-dir --default-timeout=300 --retries 1 -r requirements.txt

COPY . /model

VOLUME [ "/data" ]
VOLUME [ "/output" ]

ENV INPUT=""
ENV OUTPUT=""

CMD python3 -m DynaSD_scalp "/data/${INPUT}" "/output/${OUTPUT}"