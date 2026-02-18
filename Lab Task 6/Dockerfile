# Use official Python runtime as a parent image
FROM python:3.11-slim

# set workdir
WORKDIR /app

# copy project
COPY . /app

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# expose port
EXPOSE 5000

# run using gunicorn
CMD ["gunicorn", "run:app", "-b", "0.0.0.0:5000", "--workers", "2"]
