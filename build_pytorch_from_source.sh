from pathlib import Path

script_path = Path("/mnt/data/build_pytorch_from_source.sh")
script_content = """#!/bin/bash
set -e

# PyTorch build script with CUDA 12.8 and cuDNN 8.9.7 for RTX 5080 (sm_120)

# Step 1: Clone the repo (if not already)
# git clone --recursive https://github.com/pytorch/pytorch.git
# cd pytorch

# Step 2: Checkout correct commit
git reset --hard
git clean -fdx
git checkout 08f5371
git submodule sync
git submodule update --init --recursive

# Step 3: Export environment variables
export TORCH_CUDA_ARCH_LIST="8.9+PTX;12.0"
export USE_CUDA=1
export USE_CUDNN=1
export USE_MKLDNN=0
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Step 4: Clean any prior builds
python setup.py clean
rm -rf build dist torch.egg-info CMakeCache.txt CMakeFiles
find . -name CMakeCache.txt -delete
find . -name CMakeFiles -type d -exec rm -rf {} +
find . -name build.ninja -delete
find . -name Makefile -delete

# Step 5: Build PyTorch wheel
python setup.py bdist_wheel

# Step 6: Install the built wheel (optional)
# cd dist
# pip install torch-2.8.0a0+git08f5371-*.whl
"""

# Save the script
script_path.write_text(script_content)
script_path.chmod(0o755)

script_path.name
