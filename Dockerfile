# Optimized ClimatePredict AI Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Streamlit specific environment variables
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
ENV STREAMLIT_SERVER_MAX_MESSAGE_SIZE=200
ENV STREAMLIT_SERVER_ENABLE_CORS=true
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy optimized requirements
COPY requirements.txt .

# Install Python dependencies with specific versions
RUN pip install --no-cache-dir --upgrade pip==23.3.1 && \
    pip install --no-cache-dir --timeout 60 -r requirements.txt && \
    pip cache purge

# Copy application files
COPY climate_ai_app_enhanced.py .
COPY floating_chatbot.py .

# Create .streamlit directory (config handled via CMD arguments)
RUN mkdir -p .streamlit

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8501

# Extended health check for AWS
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Optimized startup command with mobile support
CMD ["streamlit", "run", "climate_ai_app_enhanced.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false", \
     "--server.maxUploadSize=200", \
     "--server.maxMessageSize=200", \
     "--server.enableCORS=true", \
     "--server.enableXsrfProtection=false", \
     "--server.fileWatcherType=none", \
     "--server.runOnSave=false"]