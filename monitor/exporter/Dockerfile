FROM python:3.9.6
WORKDIR /app
COPY exporter.py requirements.txt ./
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["python", "exporter.py", "--rate", "30"]
