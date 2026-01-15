#!/bin/bash
# Launch script for ETL Configuration Builder Web Interface

echo "======================================================================"
echo "  ETL Configuration Builder - Web Interface"
echo "======================================================================"
echo ""
echo "Starting Flask server..."
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install requirements if needed
if ! python -c "import flask" 2>/dev/null; then
    echo "Installing required packages..."
    pip install -r requirements-web.txt
fi

# Launch the application
python etl_config_web.py
