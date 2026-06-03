# Imagen base de Python compatible con TensorFlow
FROM python:3.10-slim

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY backend/requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el proyecto
COPY backend /app/backend
COPY frontend /app/frontend

# Entrar al backend
WORKDIR /app/backend

# Puerto que Render asignará dinámicamente
EXPOSE 10000

# Iniciar FastAPI
CMD ["python", "main.py"]