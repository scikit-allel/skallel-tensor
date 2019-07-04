from .version import version as __version__  # noqa


from .api import foo

try:
    from . import numpy_backend
except ImportError:
    pass
try:
    from . import dask_backend
except ImportError:
    pass
