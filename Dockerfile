# Dockerfile

FROM python:3.9
RUN apt-get update
RUN apt-get install nano

RUN mkdir wd
WORKDIR wd
RUN pip3 install '.[gui]'

COPY ./ ./

CMD [ "gunicorn", "--workers=1", "-b 0.0.0.0:80", "--threads=1", "gui/application:server"]
