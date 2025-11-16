FROM python:3.13.5-slim-bookworm

# Add uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first (for Docker caching)
COPY pyproject.toml uv.lock ./

# Install dependencies into /app/.venv
RUN uv sync --locked

# Ensure virtualenv binaries are used
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Copy entire project
COPY . .

EXPOSE 9696

# Run FastAPI app
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9696", "--reload"]