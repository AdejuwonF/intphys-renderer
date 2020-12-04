import enum
from enum import Enum

# Shorter, but disabled, since then Pycharm autocompletion won't work
# Backend = Enum('Backend', 'CORA AI2THOR_MCS')  # TDW


@enum.unique
class Backend(Enum):
    CORA = 0
    AI2THOR_MCS = 1
    # TDW = 2
