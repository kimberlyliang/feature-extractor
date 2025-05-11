# FROM python:3.12

# WORKDIR /

# COPY requirements.txt /app/requirements.txt

# RUN pip install -r /app/requirements.txt

# COPY app/ /app

# CMD ["python3.12", "/app/main.py"]
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install -r requirements.txt
COPY app/ /app/
COPY .env /app/.env
RUN mkdir -p data/input data/output
ENV INPUT_DIR=/data/input
ENV OUTPUT_DIR=/data/output
CMD ["python", "features_univariate.py"]
