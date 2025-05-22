FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 10000
ENV PYTHONPATH /app
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8000} app:app"]