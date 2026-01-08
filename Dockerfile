# --- base image ---
FROM python:3.11-slim

# --- environment variables ---
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# --- set workdir ---
WORKDIR /app

# --- install system dependencies ---
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# --- install pip requirements ---
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# --- copy project files ---
COPY . .

# --- Expose port for Streamlit ---
EXPOSE 8501

# --- download or cache LLM models (optional) ---
# This is optional if you want to preload GPT-J or other models
# RUN python - << 'EOF'
# from transformers import AutoModelForCausalLM, AutoTokenizer
# model_name = "EleutherAI/gpt-j-6B"
# AutoTokenizer.from_pretrained(model_name)
# AutoModelForCausalLM.from_pretrained(model_name)
# EOF

# --- Entrypoint for Streamlit app ---
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
