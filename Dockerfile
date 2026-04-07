FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install fastapi uvicorn pydantic openai requests

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]


