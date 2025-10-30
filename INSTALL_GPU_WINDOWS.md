# GPU Installation Guide for Windows

This guide will help you set up GPU acceleration for the Activity Planner on Windows.

## Prerequisites

Before installing, make sure you have:

1. **NVIDIA GPU** with CUDA support (GTX 10-series or newer, RTX series recommended)
2. **NVIDIA GPU Drivers** (latest version recommended)
   - Download from: https://www.nvidia.com/Download/index.aspx
3. **Anaconda or Miniconda** installed
   - Download from: https://www.anaconda.com/download or https://docs.conda.io/en/latest/miniconda.html

## Check Your GPU

Open Command Prompt or PowerShell and run:

```bash
nvidia-smi
```

You should see your GPU listed with driver version and CUDA version.

## Installation Steps

### Option 1: Using Conda (Recommended for Windows)

1. **Create a new conda environment:**

```bash
conda create -n activity-planner python=3.10
conda activate activity-planner
```

2. **Install PyTorch with CUDA support:**

```bash
# For CUDA 11.8 (recommended for most GPUs)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# OR for CUDA 12.1 (for newer GPUs)
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

3. **Install FAISS-GPU:**

```bash
conda install -c pytorch -c nvidia faiss-gpu=1.7.4
```

4. **Install other dependencies:**

```bash
pip install pandas numpy matplotlib seaborn rank-bm25 sentence-transformers jupyter ipykernel
```

5. **Verify GPU installation:**

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import faiss; print(f'FAISS GPU available: {hasattr(faiss, \"index_gpu_to_cpu\")}')"
```

### Option 2: Using Pip (Alternative)

**Note:** This method may not work on all Windows systems. Conda is recommended.

```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install pandas numpy matplotlib seaborn rank-bm25 sentence-transformers jupyter ipykernel

# Try installing faiss-gpu (may not work on Windows)
pip install faiss-gpu
```

## Verify Installation

Create a test file `test_gpu.py`:

```python
import torch
import faiss
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test FAISS GPU
print(f"\nFAISS version: {faiss.__version__}")

# Try creating a GPU index
try:
    # Check if GPU resources are available
    res = faiss.StandardGpuResources()

    # Create a simple index
    d = 128  # dimension
    index_cpu = faiss.IndexFlatL2(d)

    # Move to GPU
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)

    # Add some vectors
    xb = np.random.random((1000, d)).astype('float32')
    index_gpu.add(xb)

    print(f"✓ FAISS GPU is working! Index has {index_gpu.ntotal} vectors")
except Exception as e:
    print(f"✗ FAISS GPU not available: {e}")
    print("  Falling back to CPU mode")
```

Run it:

```bash
python test_gpu.py
```

## Using GPU in Jupyter Notebook

1. **Register your conda environment with Jupyter:**

```bash
conda activate activity-planner
python -m ipykernel install --user --name=activity-planner --display-name "Python (Activity Planner GPU)"
```

2. **Start Jupyter:**

```bash
jupyter notebook
```

3. **Select the kernel:**
   - In Jupyter, go to Kernel → Change kernel → Python (Activity Planner GPU)

## Troubleshooting

### Error: "CUDA driver version is insufficient"

- Update your NVIDIA GPU drivers to the latest version
- Make sure your GPU supports the CUDA version you're trying to install

### Error: "Could not load dynamic library 'cudart64_*.dll'"

- Install or reinstall CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads
- Make sure CUDA bin directory is in your PATH environment variable

### FAISS-GPU installation fails

- Use conda instead of pip (recommended for Windows)
- Make sure you have CUDA toolkit installed
- Try installing an older version: `conda install faiss-gpu=1.7.3`

### GPU is not being used

- Check that CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Make sure you're using the GPU-enabled code in the notebook
- The notebook will automatically detect and use GPU if available

## Performance Benefits

With GPU acceleration:
- **FAISS indexing:** 5-20x faster (depending on dataset size)
- **Similarity search:** 10-100x faster (especially for large datasets)
- **Sentence-BERT encoding:** 3-5x faster with GPU-enabled PyTorch

For the activity planner with ~100-1000 activities, you'll see noticeable improvements in search response time.

## Fallback to CPU

If GPU setup doesn't work, you can always fall back to CPU mode:

```bash
pip install -r requirements.txt
```

The notebook will automatically detect whether GPU is available and use the best option.
