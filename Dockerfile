# Используем CUDA-базовый образ для GPU
FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04

# Устанавливаем базовые утилиты
RUN apt-get update && apt-get install -y python3-pip

# Рабочая директория
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем остальные зависимости
RUN pip3 install --no-cache-dir -r requirements.txt --break-system-packages

# Копируем весь проект
COPY . .

# По умолчанию запускаем gen.py
ENTRYPOINT ["python3", "gen.py", "--config", "configs/config.json"]

