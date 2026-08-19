# Image de reproduction : environnement figé par uv.lock, tests et expériences légères.
# Construction : docker build -t zonefleet .
# Vérification : docker run --rm zonefleet make check
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    NML_DISABLE_USAGE_LOGGING=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends make libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --all-extras

COPY . .
CMD ["make", "check"]
