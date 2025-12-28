#!/bin/bash
#SBATCH --job-name=kan_benchmark_grace
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=../results/slurm_%j.out
#SBATCH --error=../results/slurm_%j.err
#SBATCH --mail-type=END,FAIL

# ============================================================================
# KAN vs MLP vs XGBoost Benchmark - Yale Grace Cluster
# ============================================================================
# Estimated runtime: ~1-2 hours
# 
# Usage:
#   1. Transfer the project to Grace (run on laptop):
#      rsync -avz --exclude '.git' --exclude 'results' \
#        "/Users/matthewsahagun/Downloads/Kolmogorov-Arnold Network Project/" \
#        ms4726@grace.ycrc.yale.edu:~/kan_benchmark/
#   
#   2. SSH into Grace:
#      ssh ms4726@grace.ycrc.yale.edu
#   
#   3. Navigate to scripts folder and submit:
#      cd ~/kan_benchmark/scripts
#      sbatch submit_grace.sh
#   
#   4. Monitor job:
#      squeue -u ms4726
#      tail -f ../results/slurm_*.out
#   
#   5. Download results when complete (run on laptop):
#      scp -r ms4726@grace.ycrc.yale.edu:~/kan_benchmark/results ./
# ============================================================================

echo "=============================================="
echo "KAN Benchmark Job Started (Grace)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo ""

# Load required modules
# Grace typically uses miniconda or miniforge. 
# Adjust 'module load' if your user profile handles conda differently.
module purge
module load miniconda
module load CUDA/12.1

# Activate conda environment (create if needed)
if ! conda env list | grep -q "kan_benchmark"; then
    echo "Creating conda environment 'kan_benchmark'..."
    conda create -n kan_benchmark python=3.10 -y
fi

source activate kan_benchmark

# Install dependencies if not already installed
echo "Installing dependencies..."
# Ensure pip is up to date
pip install --upgrade pip
# Install torch compatible with CUDA 12.1
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q -r requirements.txt

# Verify GPU availability
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Create results directory inside the job to ensure it exists
mkdir -p ../results/figures

# Run the benchmark
echo "=============================================="
echo "Starting Benchmark..."
echo "=============================================="

python -u run_benchmark.py

echo ""
echo "=============================================="
echo "Job Completed"
echo "=============================================="
echo "End time: $(date)"
echo "Results saved to: ../results/"
echo ""
