FROM python:3.11
LABEL authors="d1ffic00lt"

COPY ../api ./api
COPY ../parser ./api/parser
COPY ../preprocessing ./api/preprocessing

WORKDIR /api

RUN pip install tinkoff-investments
RUN pip install -r requirements.txt

CMD ["python", "main.py"]
