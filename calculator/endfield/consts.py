from enum import Enum


class EndfieldBannerType(Enum):
    CHARA = 0
    WEAPON = 1

class EndfieldResourceAction(Enum):
    ADJUST_DUMMY = 0

class EndfieldCharaPullResultType(Enum):
    GET_NOTHING = 0
    GET_PITY = 1
    GET_TARGET = 2
    TERMINAL_STATE = 3