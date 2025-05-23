FROM python:3.10-slim
WORKDIR /app

# Install Chromium and its dependencies
RUN apt-get update && apt-get install -y \
    chromium \
    libnss3 \
    libgconf-2-4 \
    libxi6 \
    libxrender1 \
    libxtst6 \
    libfontconfig1 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code into the container
COPY . .

EXPOSE 8000
ENV PYTHONPATH /app

CMD ["gunicorn", "-w", "3", "-b", "0.0.0.0:8000", "app:app"] 