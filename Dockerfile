FROM python:3.10-slim

# Create cache folder
RUN mkdir -p /app/cache && chmod -R 777 /app/cache

WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
