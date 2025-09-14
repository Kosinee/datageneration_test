# Используем CUDA-базовый образ для GPU
FROM nvidia/cuda:12.1.105-runtime-ubuntu22.04

# Устанавливаем базовые утилиты
RUN apt-get update && apt-get install -y python3-pip

# Рабочая директория
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем pip зависимости
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Копируем весь проект
COPY . .

# По умолчанию запускаем gen.py
ENTRYPOINT ["python3", "gen.py", "--config", "configs/config.json"]

