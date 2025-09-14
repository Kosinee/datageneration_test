FROM python:3.10

WORKDIR /app

COPY requirements.txt .

# Install system dependencies
RUN apt-get update && pip install --no-cache-dir -r requirements.txt

COPY . .

# Run the ingestion script
ENTRYPOINT ["python", "gen.py", "--config", "configs/config.json"]
