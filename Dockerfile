FROM python:3.9-slim AS base

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt 

COPY . .
COPY .env .


EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app_v2.py"]
