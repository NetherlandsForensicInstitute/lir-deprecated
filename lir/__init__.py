"""
Experimental sub package providing multiclass analysis, a generalization of the
main package which provides only biclass analysis. The API is as close as
possible to the main package.

WARNING: The multiclass package is experimental and far from complete.

This package may be used as a substitute for the main package:
```
import lir.multiclass as lir
```

Or they can be used side by side:
```
import lir as bilir
import lir as multilir
```
"""

from . import metrics
from .lr import *
from .calibration import *
from .generators import *
from .util import Xn_to_Xy, Xy_to_Xn
from . import plotting
