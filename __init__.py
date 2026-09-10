# Postprocessing python scripts for KORAL hdf5 output
#
# Makes this directory importable as a package from elsewhere:
#
#   import sys; sys.path.append('/path/to/parent/of/koral_hdf5_postproc')
#   from koral_hdf5_postproc import read_koral_hdf52D, gridprofile
#
# The scripts still run standalone (python avgkoralhdf5s.py avg DIR ...).

from . import metricKS
from . import koralopacities
from . import koralh5postproc
from . import avgkoralhdf5s

# reading and plotting phi-reduced (2D) and full (3D) files
from .koralh5postproc import (simdata2D, simdata3D,
                              read_koral_hdf52D, read_koral_hdf53D,
                              gridprofile)

# phi-averaging, phi-slicing and time-averaging
from .avgkoralhdf5s import (phireduce, phireduce_hdf5, tavg_hdf5s,
                            peek_time, is_3d_hdf5)
