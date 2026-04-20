## SSH into Northeastern HPC
```sh
ssh user.name@login.explorer.northeastern.edu
# use your northastern credentials

# if necessary, hop to the explorer node with previous tmux session
ssh explorer-01 # if you are not on 01 but have a tmux session there, hop to 01 first, then tmux attach
```

## Requesting GPU resources on Northeastern HPC
- Check gpus available on the cluster
```sh
sinfo -p gpu --Format=nodes,cpus,memory,features,statecompact,nodelist,gres
sinfo -p courses-gpu --Format=nodes,cpus,memory,features,statecompact,nodelist,gres
```
- Request a GPU node with an interactive session (adjust time, memory, and CPU as needed)
```sh
srun -p courses-gpu --gres=gpu:1 --time=4:00:00 --mem=32GB --cpus-per-task=4 --pty /bin/bash
```

## General commands for a new HPC session
Do use `tmux` whenever you ask the HPC to run something for you, so that you can safely disconnect without killing your job.
```sh
# Set Cache boundaries to protect Home Quota
export HF_HOME=/scratch/$USER/huggingface_cache
export TRANSFORMERS_CACHE=/scratch/$USER/huggingface_cache
export CONDA_PKGS_DIRS=/scratch/$USER/conda_pkgs
export PIP_CACHE_DIR=/scratch/$USER/pip_cache

# Load system environments (Crucial order: purge first, THEN load explorer for proxies)
module purge
module load explorer anaconda3/2024.06 cuda/12.1.1

# Activate existing Python environment
source activate /scratch/$USER/6120_env

# Run your script
cd /home/xu.x1/6120/RAG/
# my home contains this git repo
git pull
python
```

## Command for saving dataset into vector database (FAISS)
```sh
python build_index.py
```

## Command for running the Streamlit frontend
```sh
streamlit run frontend.py --server.port 8501 --server.headless true
```

## Rebuild /scratch/$USER/6120_env if you need to

```sh
# 1. Setup Caches to protect Home Quota
export HF_HOME=/scratch/$USER/huggingface_cache
export TRANSFORMERS_CACHE=/scratch/$USER/huggingface_cache
export CONDA_PKGS_DIRS=/scratch/$USER/conda_pkgs
export PIP_CACHE_DIR=/scratch/$USER/pip_cache

# 2. Re-create directories (just in case they were also wiped)
mkdir -p /scratch/$USER/huggingface_cache
mkdir -p /scratch/$USER/conda_pkgs
mkdir -p /scratch/$USER/pip_cache

# 3. Load Modules (Critical: explorer handles the proxy, cuda handles the GPU)
module purge
module load explorer anaconda3/2024.06 cuda/12.1.1

# 4. Create the new Conda Environment
conda create --prefix /scratch/$USER/6120_env -c conda-forge python=3.12.4 -y

# 5. Activate the new empty environment
source activate /scratch/$USER/6120_env

# 6. Install project dependencies from the repo
cd /home/xu.x1/6120/RAG/
pip install -r requirements.txt

# 7. Reinstall exactly what the HPC needs for GPU acceleration (PyTorch on CUDA 12.1)
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 8. Run your script!
python build_index.py
```