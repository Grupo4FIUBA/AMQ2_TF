# Usar una imagen base liviana de Python
FROM python:3.10-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos al contenedor
COPY ./requirements.txt ./requirements.txt
COPY ./app.py ./app.py
COPY ./s3.py ./s3.py


# Instalar las dependencias
RUN pip install --upgrade pip && pip install -r requirements.txt

# Exponer el puerto por defecto de FastAPI/Uvicorn
EXPOSE 8000

# Comando para ejecutar la app con Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]