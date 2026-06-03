# 1. Usar una imagen oficial de Python ligera
FROM python:3.10-slim

# 2. Configurar la carpeta de trabajo dentro del servidor
WORKDIR /app

# 3. Instalar las librerías necesarias de golpe
RUN pip install --no-cache-dir fastapi uvicorn python-multipart keras tensorflow-cpu pillow numpy

# 4. Copiar los archivos de tu proyecto al servidor
COPY ./backend /app/backend
COPY ./frontend /app/frontend

# 5. Movernos a la carpeta del backend para ejecutar
WORKDIR /app/backend

# Exponer el puerto nativo que usa Hugging Face
EXPOSE 7860

# 6. Comando para arrancar el servidor de FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]