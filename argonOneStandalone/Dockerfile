# hadolint ignore=DL3006
FROM python:3-alpine

RUN  apk add i2c-tools --no-cache;

WORKDIR /opt/argonone

COPY requirements.txt /opt/argonone/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./app /opt/argonone/app

CMD ["fastapi", "run", "app/main.py", "--port", "8023"]


