FROM python:3.13
LABEL authors="d1ffic00lt"
WORKDIR /web

COPY . .

RUN pip install -r requirements.txt

CMD ["streamlit", "run", "app.py"]

# STREAMLIT_SERVER_PORT
# STREAMLIT_SERVER_HEADLESS