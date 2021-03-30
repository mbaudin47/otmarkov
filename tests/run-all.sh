#!/bin/sh

set -xe

echo "Python interpreter"
echo `which python`
echo "OpenTURNS version"
python -c "import openturns; print(openturns.__version__); exit()"

# Run tests
cd ..

# Demos
cd demos
python calcul-exact.py
python calcul-PDMP-basic.py
python calcul-PDMP-low-discrepancy.py
cd ..

# Unit tests
cd tests
python -m unittest discover .
cd ..

# Notebooks in all subdirectories
python tests/find-ipynb-files.py
