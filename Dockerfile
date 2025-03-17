FROM  python:3.9-slim-buster
   WORKDIR /app
   COPY . /app
   RUN apt update -y && apt install awscli -y
   RUN pip install --upgrade pip

   RUN pip install --timeout=500  -r requirements.txt
   CMD ["python3", "app.py"]