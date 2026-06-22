FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY . .
CMD ["python", "-m", "uvicorn", "main:http_app", "--host", "0.0.0.0", "--port", "8000"]
