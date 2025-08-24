FROM python:3.12-bookworm
WORKDIR /usr/src/app
COPY requirements.txt ./
COPY . .

# Install system dependencies needed for Playwright
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv myenv && \
    . myenv/bin/activate && \
    pip install --no-cache-dir --upgrade -r ./requirements.txt && \
    pip install playwright && \
    playwright install chromium

# Activate virtual environment in CMD
ENV PATH="/usr/src/app/myenv/bin:$PATH"

EXPOSE 8000
CMD ["uvicorn", "src.carver:app", "--port", "8000", "--host", "0.0.0.0"]
