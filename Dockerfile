FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install .

COPY config ./config
COPY data ./data
COPY deploy ./deploy
COPY pipelines ./pipelines

# OpenShift runs containers as an arbitrary UID in group 0, so group 0 gets the owner's permissions.
RUN useradd --uid 1001 --gid 0 --create-home app && chown -R 1001:0 /app && chmod -R g=u /app
USER 1001
EXPOSE 8080
CMD ["uvicorn", "--factory", "fieldassist.api:create_app", "--host", "0.0.0.0", "--port", "8080"]
