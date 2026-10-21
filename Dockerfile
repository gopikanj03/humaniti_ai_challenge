# ClassCritique on Hugging Face Spaces (Docker SDK). Secrets (GROQ_API_KEY, HOSTED_PROVIDER_APPROVED) are set
# as Space secrets and arrive as environment variables; nothing secret is in this image.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run containers as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .
# Fetch the speaker-embedding model at build time (checked by SHA-256) so the first analysis isn't slower.
RUN python -c "from src.models import ensure_model; ensure_model('speaker_embedding')"

EXPOSE 7860
# XSRF/CORS checks are relaxed because Spaces serve the app inside an iframe; without this, uploads fail.
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false"]
