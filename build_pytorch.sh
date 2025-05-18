#!/bin/bash
set -e

echo "🚀 Starting PyTorch build..."

# 1. Set required environment flags
export CFLAGS="-mcmodel=large"
export CXXFLAGS="-mcmodel=large"
export LDFLAGS="-mcmodel=large"

# 2. Enable large swap (if not already enabled)
if ! grep -q "/swapfile" /etc/fstab; then
    echo "🧠 Creating 64GiB swap..."
    sudo fallocate -l 64G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
else
    echo "✅ Swap already set up."
fi

# 3. Navigate to PyTorch root and clean any prior build
cd ~/pytorch
rm -rf build
mkdir -p build && cd build

# 4. Run cmake with flags and Ninja generator
echo "⚙️  Running CMake..."
cmake .. \
  -DCMAKE_C_FLAGS="$CFLAGS" \
  -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
  -DCMAKE_SHARED_LINKER_FLAGS="$LDFLAGS" \
  -GNinja \
  -DCMAKE_BUILD_TYPE=Release

# 5. Build with ninja and log output
echo "🛠️  Building with Ninja..."
ninja | tee build.log

echo "🎉 Build finished successfully!"
