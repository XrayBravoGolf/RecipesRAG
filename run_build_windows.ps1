$ErrorActionPreference = "Stop"

Write-Host "Checking for Python..."
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not installed or not in PATH. Please install Python first."
}

Write-Host "Creating virtual environment (.venv)..."
python -m venv .venv

Write-Host "Activating virtual environment..."
& .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing PyTorch with CUDA support (cu121)..."
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host "Installing other dependencies..."
python -m pip install sentence-transformers faiss-cpu pyarrow tqdm

Write-Host "Starting index build. This might take a while..."
python build_index.py

Write-Host "Build complete! You can now copy the artifacts/index directory back to your Linux machine."
