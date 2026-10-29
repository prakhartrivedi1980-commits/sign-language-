
import sys
import numpy
import matplotlib
import pandas
import sklearn
import pytest
import yaml
import tqdm

print("=" * 45)
print("SIGN LANGUAGE ASSISTANT")
print("DEPENDENCY VERIFICATION")
print("=" * 45)

print("Python:", sys.version.split()[0])
print("NumPy:", numpy.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Pandas:", pandas.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Pytest:", pytest.__version__)
print("PyYAML:", yaml.__version__)
print("TQDM:", tqdm.__version__)

print("\nAll Phase 1 dependencies installed successfully!")
