# This script builds the Python project.

# 1. Create a virtual environment in a directory named '.venv'
#python -m venv .venv

# 2. Activate the virtual environment
#. .\.venv\Scripts\Activate.ps1

# 3. Install dependencies from requirements.txt
uv pip install -r requirements.txt

# 4. Install the 'build' package
uv pip install build

# 5. Run the build process
python -m build

echo "Build complete. Find the artifacts in the 'dist' directory."
