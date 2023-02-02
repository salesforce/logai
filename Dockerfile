# Dockerfile

FROM python:3.9-slim

# Create work dir
RUN mkdir wd
WORKDIR wd

COPY . .

# Setup env
RUN apt-get update
RUN apt-get install nano

# Install dependencies
RUN pip install --upgrade pip gunicorn
RUN pip install ".[all]"
RUN python -m nltk.downloader

# Setup PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:./"
# Run locally on port 80
CMD [ "gunicorn", "--workers=3", "-b 0.0.0.0:80", "--threads=1", "gui.application:server"]
