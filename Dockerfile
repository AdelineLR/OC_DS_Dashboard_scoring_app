FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y libgomp1 && \
    apt-get clean
    
# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY app/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the dashboard code (the 'app' folder) into the container
COPY app /app/app

# Expose port 8501 to access the dashboard
EXPOSE 8501

# Command to run the dashboard
ENTRYPOINT ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]