# Import all required packages
# Flattens the importtime profile
# These take up ~92% of cold import time
import numpy
import scipy
import scipy.stats

# Import the actual Study object
from .study import Study

# Expose dynamic version
import importlib.metadata

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback for development mode
