FROM python:3.12-slim

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 1. Install dependencies first
RUN uv sync --frozen --no-install-project

# 2. Copy the rest of the application code
COPY . .

# 3. Create a non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Start command
CMD ["uv", "run", "python", "main.py"]
