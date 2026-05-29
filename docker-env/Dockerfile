# Start from the official slim Python base image
FROM python:3.10-slim

# Set a working directory inside the image
WORKDIR /usr/src/app

# Copy the requirements file into the image
COPY requirements.txt ./
COPY wdnets/ ./wdnets/

# Install the Python dependencies
# --no-cache-dir makes the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# This part is just metadata; the entrypoint will be overridden by your Python script
CMD [ "python" ]