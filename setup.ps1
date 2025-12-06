# Create Python virtual environment
Write-Host "Creating Python virtual environment..."
python -m venv venv

# Activate virtual environment and install dependencies
Write-Host "Activating virtual environment and installing dependencies..."
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

Write-Host "Setup complete. You can now run the analysis with 'python reproduce_analysis.py'"
