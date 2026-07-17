#!/bin/bash
# ConsumerGuard AI — Azure App Service Startup Script
# This file tells Azure how to launch the Streamlit application.
#
# Azure App Service calls this script on startup.
# The app must listen on port 8000 (Azure's default port for custom apps).

# Activate the virtual environment created by Azure Oryx build
if [ -d "/home/site/wwwroot/antenv" ]; then
    source /home/site/wwwroot/antenv/bin/activate
elif [ -d "antenv" ]; then
    source antenv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

streamlit run app.py \
    --server.port 8000 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
