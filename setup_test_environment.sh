#!/bin/bash
# Setup script for model testing environment

echo "========================================="
echo "Activity Planner - Model Testing Setup"
echo "========================================="
echo

# Check Python version
echo "Checking Python version..."
python --version
echo

# Install requirements
echo "Installing required packages..."
pip install -r requirements.txt

echo
echo "Installing additional testing packages..."
pip install matplotlib seaborn

echo
echo "Verifying installation..."
python -c "
import sys
try:
    import torch
    print('✓ PyTorch:', torch.__version__)
except ImportError:
    print('✗ PyTorch not found')
    sys.exit(1)

try:
    import sklearn
    print('✓ scikit-learn:', sklearn.__version__)
except ImportError:
    print('✗ scikit-learn not found')
    sys.exit(1)

try:
    import matplotlib
    print('✓ matplotlib:', matplotlib.__version__)
except ImportError:
    print('✗ matplotlib not found')
    sys.exit(1)

try:
    import seaborn
    print('✓ seaborn:', seaborn.__version__)
except ImportError:
    print('✗ seaborn not found')
    sys.exit(1)

try:
    import sentence_transformers
    print('✓ sentence-transformers:', sentence_transformers.__version__)
except ImportError:
    print('✗ sentence-transformers not found')
    sys.exit(1)

print()
print('All required packages are installed!')
"

echo
echo "========================================="
echo "Setup complete! You can now run:"
echo "  python test_models.py"
echo "========================================="
