-- Create a dedicated database for the MLflow backend store.
-- Runs automatically on first Postgres init (docker-entrypoint-initdb.d).
SELECT 'CREATE DATABASE mlflow_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow_db')\gexec
