FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
  && update-ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --upgrade --force-reinstall openai==1.12.0 httpx==0.26.0

RUN pip install --no-cache-dir -U yt-dlp

RUN python3 -m pip install --upgrade certifi && \
    ln -sf "$(python3 -c 'import certifi; print(certifi.where())')" /etc/ssl/certs/ca-certificates.crt

COPY . .

CMD ["python", "-m", "bot.main"]
