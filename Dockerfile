FROM python:3.12-slim

# System deps needed by Azure Cognitive Services Speech SDK (and audio)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libuuid1 \
        libasound2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "pipecat_twilio_bot.py"]


