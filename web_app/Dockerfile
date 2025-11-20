FROM python:3.10-slim

ENV POPPLER_PATH=/usr/bin

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        poppler-utils \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY web_app/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy application
COPY web_app /app

COPY runs /app/runs
COPY runs /runs

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD curl -f http://localhost:7860/ || exit 1

CMD ["uvicorn", "web_app.app:app", "--host", "0.0.0.0", "--port", "7860"]
