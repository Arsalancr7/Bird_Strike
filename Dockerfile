FROM python:3.10-slim

WORKDIR /app

# Install dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the application files and required folders
COPY app.py .
COPY models/ ./models/
COPY vectorstore/ ./vectorstore/

EXPOSE 8080
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]


