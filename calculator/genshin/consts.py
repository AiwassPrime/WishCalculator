from enum import Enum


class GenshinBannerType(Enum):
    CHARA = 0
    WEAPON = 1


class GenshinCharaPityType(Enum):
    CHARA_50 = 0
    CHARA_100 = 1


class GenshinWeaponPityType(Enum):
    WEAPON_50_PATH_0 = 0
    WEAPON_100_PATH_0 = 1
    WEAPON_50_PATH_1 = 2
    WEAPON_100_PATH_1 = 3
    WEAPON_50_PATH_2 = 4
    WEAPON_100_PATH_2 = 5


class GenshinResourceAction(Enum):
    ADJUST_FATE = 0
    ADJUST_CRYSTAL = 1
    ADJUST_GEM = 2
    ADJUST_STAR = 3
    FROM_STAR_TO_FATE = 4
    FROM_GEM_TO_FATE = 5
    FROM_CRYSTAL_TO_GEM = 6


class GenshinPullResultType(Enum):
    GET_NOTHING = 0
    GET_PITY = 1
    GET_NON_TARGET = 2
    GET_TARGET = 3
    TERMINAL_STATE = 4
