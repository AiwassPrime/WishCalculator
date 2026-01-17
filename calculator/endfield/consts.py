from enum import Enum


class EndfieldCharaPullResultType(Enum):
    GET_NOTHING = 0
    GET_PITY = 1
    GET_TARGET = 2
    TERMINAL_STATE = 3