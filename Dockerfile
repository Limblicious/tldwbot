FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --upgrade --force-reinstall openai==1.12.0 httpx==0.26.0

COPY . .

CMD ["python", "-m", "bot.main"]
