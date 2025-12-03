# Use the official Ubuntu 22.04 base image
FROM ubuntu:22.04

# Install Python 3.10 and necessary build dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set 'python3.10' as the default 'python' executable
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# --- FIX: Upgrade pip to resolve the Assertion Error in the dependency resolver ---
RUN pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
# Install pymilvus first, using the fixed pip
RUN pip install "pymilvus[milvus_lite]"
# Install main dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY ./app/papers.csv /app/papers.csv
COPY ./app/milvus_demo.db /app/milvus_demo.db
COPY ./app/finetuned_model_from_analysis /app/finetuned_model_from_analysis/

# Copy the application code
COPY app.py .

# Define environment variables
ENV PORT=8080
ENV INPUT_CSV_PATH="/app/papers.csv"
ENV MILVUS_DB_PATH="/app/milvus_demo.db"
ENV MODEL_NAME="all-MiniLM-L6-v2"
ENV FINETUNED_MODEL_PATH="/app/finetuned_model_from_analysis"

# Command to run the application
#CMD ["python", "app.py"]
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app


