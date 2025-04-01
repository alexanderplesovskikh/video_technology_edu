FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем сам скрипт
COPY worker_videoedu.py .

# Команда для запуска скрипта
CMD ["python", "worker.py"]
