# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

FROM python:3.13-slim

# Install OpenJDK 17, Tesseract OCR, and some language packs (https://packages.debian.org/sid/tesseract-ocr-language)
RUN apt-get update && \
    apt-get install -y wget openjdk-17-jre tesseract-ocr tesseract-ocr-eng tesseract-ocr-fra && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download Tika Server for offline use
# https://github.com/chrismattmann/tika-python?tab=readme-ov-file#airgap-environment-setup
RUN wget https://repo1.maven.org/maven2/org/apache/tika/tika-server-standard/3.1.0/tika-server-standard-3.1.0.jar -O /tmp/tika-server.jar && \
    wget https://repo1.maven.org/maven2/org/apache/tika/tika-server-standard/3.1.0/tika-server-standard-3.1.0.jar.md5 -O /tmp/tika-server.jar.md5
ENV TIKA_SERVER_JAR=/tmp/tika-server.jar

# Set working directory
WORKDIR /app

# Copy application files (excluding tests)
COPY scanner.py /app/
COPY setup.py /app/
COPY cli/ /app/cli/
COPY classifyit/ /app/classifyit/
COPY content_categories /app/content_categories

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Set the entrypoint
ENTRYPOINT ["python", "scanner.py"]