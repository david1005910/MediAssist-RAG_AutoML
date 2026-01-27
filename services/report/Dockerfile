FROM python:3.11-slim

WORKDIR /app

# Install build tools and dependencies for wkhtmltopdf
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    fontconfig \
    libfreetype6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libjpeg62-turbo \
    libpng16-16 \
    xfonts-75dpi \
    xfonts-base \
    && rm -rf /var/lib/apt/lists/* \
    # wkhtmltopdf is optional - if not available, PDF generation will use fallback
    || true

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
