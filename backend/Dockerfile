FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DATABASE_PATH=/data/annotations.sqlite3 \
    MODELNET10_DATA_DIR=/data/modelnet10 CLASSIFICATION_RUNS_DIR=/data/runs
COPY backend/requirements.txt ./requirements.txt
# The CPU build of PyTorch keeps the image small; the pinned torch version is the same.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt \
    && useradd --create-home viewer && mkdir /data && chown viewer:viewer /data
COPY --chown=viewer:viewer backend/app ./app
COPY --chown=viewer:viewer backend/models ./models
COPY --chown=viewer:viewer backend/samples ./samples
USER viewer
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
