FROM python:3.13-slim
WORKDIR /app
# Deterministic, offline ranking
ENV PYTHONHASHSEED=0
ENV PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "run_submission.py"]
