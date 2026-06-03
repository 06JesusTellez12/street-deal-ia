FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .

RUN pip install --upgrade pip

RUN pip install \
    --no-cache-dir \
    -r requirements.txt

COPY backend /app/backend
COPY frontend /app/frontend

WORKDIR /app/backend

EXPOSE 10000

CMD ["uvicorn","main:app","--host","0.0.0.0","--port","10000"]