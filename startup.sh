#!/bin/bash
# ConsumerGuard AI — Azure App Service Startup Script
# This file tells Azure how to launch the Streamlit application.
#
# Azure App Service calls this script on startup.
# The app must listen on port 8000 (Azure's default port for custom apps).

pip install -r requirements.txt

streamlit run app.py \
    --server.port 8000 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
