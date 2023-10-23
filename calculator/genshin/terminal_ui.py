import os

from calculator import definitions
from calculator.genshin import user, consts
from loguru import logger

SEPERATOR = "------------------------------"


def collect_user_info(input_user: user.GenshinUser):
    check = True
    while check:
        try:
            check = False
            print(SEPERATOR)
            fate = int(input("Please enter your Intertwined Fate: "))
            gem = int(input("Please enter your Primogem: "))
            crystal = int(input("Please enter your Genesis Crystal: "))
            star = int(input("Please enter your Masterless Starglitter: "))
            input_user.update_resource(consts.GenshinResourceAction.ADJUST_FATE, fate)
            input_user.update_resource(consts.GenshinResourceAction.ADJUST_GEM, gem)
            input_user.update_resource(consts.GenshinResourceAction.ADJUST_CRYSTAL, crystal)
            input_user.update_resource(consts.GenshinResourceAction.ADJUST_STAR, star)
        except ValueError:
            print(SEPERATOR + "\n" +
                  "Please input an integer")
            check = True
    print(SEPERATOR + "\n" +
          "You have {} pull(s)".format(input_user.get_total_pull()[0]))

    chara = 0
    check = True
    while check:
        try:
            check = False
            print(SEPERATOR)
            chara = int(input("Please enter the number of pulls since the last five-star character: "))
            if chara < 0 or chara >= 90:
                print(SEPERATOR + "\n" +
                      "The input is not within [0, 89]")
                check = True
                continue
        except ValueError:
            print(SEPERATOR + "\n" +
                  "Please input a valid integer")
            check = True
    chara_pity_enum = consts.GenshinCharaPityType.CHARA_50
    check = True
    while check:
        try:
            check = False
            print(SEPERATOR)
            chara_pity = int(input("0 - Next is 50/50\n"
                                   "1 - Next is guarantee\n"
                                   "Please enter your character pity status: "))
            if chara_pity == 0:
                chara_pity_enum = consts.GenshinCharaPityType.CHARA_50
            elif chara_pity == 1:
                chara_pity_enum = consts.GenshinCharaPityType.CHARA_100
            else:
                print(SEPERATOR + "\n" +
                      "Input is invalid")
                check = True
                continue
        except ValueError:
            print(SEPERATOR + "\n" +
                  "Please input a valid integer")
            check = True
    weapon = 0
    check = True
    while check:
        try:
            check = False
            print(SEPERATOR)
            weapon = int(input("Please enter the number of pulls since the last five-star weapon: "))
            if weapon < 0 or weapon >= 77:
                print(SEPERATOR + "\n" +
                      "The input is not within [0, 76]")
                check = True
                continue
        except ValueError:
            print(SEPERATOR + "\n" +
                  "Please input a valid integer")
            check = True
    weapon_pity_enum = consts.GenshinWeaponPityType.WEAPON_50_PATH_0
    check = True
    while check:
        try:
            check = False
            print(SEPERATOR)
            weapon_pity = int(input("0 - Next is 50/50 and current path is 0/2\n"
                                    "1 - Next is guarantee and current path is 0/2\n"
                                    "2 - Next is 50/50 and current path is 1/2\n"
                                    "3 - Next is guarantee and current path is 1/2\n"
                                    "4 - Next is 50/50 and current path is 2/2\n"
                                    "5 - Next is guarantee and current path is 2/2\n"
                                    "Please enter your character pity status: "))
            if weapon_pity == 0:
                weapon_pity_enum = consts.GenshinWeaponPityType.WEAPON_50_PATH_0
            elif weapon_pity == 1:
                weapon_pity_enum = consts.GenshinWeaponPityType.WEAPON_100_PATH_0
            elif weapon_pity == 2:
                weapon_pity_enum = consts.GenshinWeaponPityType.WEAPON_50_PATH_1
            elif weapon_pity == 3:
                weapon_pity_enum = consts.GenshinWeaponPityType.WEAPON_100_PATH_1
            elif weapon_pity == 4:
                weapon_pity_enum = consts.GenshinWeaponPityType.WEAPON_50_PATH_2
            elif weapon_pity == 5:
                weapon_pity_enum = consts.GenshinWeaponPityType.WEAPON_100_PATH_2
            else:
                print(SEPERATOR + "\n" +
                      "Input is invalid")
                check = True
                continue
        except ValueError:
            print(SEPERATOR + "\n" +
                  "Please input a valid integer")
            check = True
    plan = []
    check = True
    while check:
        check = False
        print(SEPERATOR)
        plan_str = input("0 - Get character\n"
                         "1 - Get weapon\n"
                         "Your input plan should be something looks like: 00010000\n"
                         "Please enter your gacha plan: ")
        for char in plan_str:
            if char == "0":
                plan.append(consts.GenshinBannerType.CHARA)
            elif char == "1":
                plan.append(consts.GenshinBannerType.WEAPON)
            else:
                plan = []
                print(SEPERATOR + "\n" +
                      "The input should be a string consist of 0 and 1")
                check = True
                break
    input_user.set_state(chara, chara_pity_enum, weapon, weapon_pity_enum, plan)
    return input_user


if __name__ == '__main__':
    logger.add("app.log", rotation="500 MB", level="INFO")

    directory_path = os.path.join(definitions.ROOT_DIR, "models")

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    is_exit = 0
    curr_user = None
    while is_exit != 1:
        if curr_user is None:
            user_account = str(input("Please enter your account: "))
            user_passcode = str(input("Please enter your password: "))
            curr_user = user.init_genshin_user(user_account, user_passcode)

        next_step = str(input(SEPERATOR + "\n" +
                              "0 - Input new user info\n"
                              "1 - Load saved data\n"
                              "2 - start a demo\n"
                              "9 - Exit\n"
                              "Please enter next step: "))
        if next_step == "9":
            is_exit = 1
            continue
        elif next_step == "0":
            curr_user = collect_user_info(curr_user)
        elif next_step == "1":
            pass
        elif next_step == "2":
            pass
        else:
            print(SEPERATOR + "\n" +
                  "Invalid input")
    print(SEPERATOR + "\n" +
          "Exit")
