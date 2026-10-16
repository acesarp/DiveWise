FROM python:3.12.9

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 3001

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:3001/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.port=3001", "--server.address=0.0.0.0"]
