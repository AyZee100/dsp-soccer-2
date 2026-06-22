FROM python:3.12-slim

WORKDIR /mlflow

# MLflow server + Postgres driver for the backend store.
RUN pip install --no-cache-dir \
    mlflow==2.16.2 \
    psycopg2-binary==2.9.9

EXPOSE 5000

# The actual `mlflow server ...` command (with credentials from env) is defined
# in docker-compose.yml so it can use ${POSTGRES_*} variables.
