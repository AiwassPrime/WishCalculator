import copy
import logging
from dataclasses import dataclass

import wish_model_v2 as model
from calculator.genshin import consts


@dataclass
class GenshinResource:
    intertwined_fate: int
    genesis_crystal: int
    primogem: int
    starglitter: int


class GenshinUser:
    def __init__(self, uid):
        self.uid = uid

        self.state = model.GenshinWishModelState(((0, 0), (0, 0, 0), []))
        self.resource = GenshinResource(0, 0, 0, 0)

        self.model = model.GenshinWishModelV2()
        self.calculator = {}

    def get_resource(self):
        logging.info("Get resources for user {}: {}".format(self.uid, self.resource))
        return self.resource

    def set_resource(self, intertwined_fate, genesis_crystal, primogem, starglitter):
        self.resource = GenshinResource(intertwined_fate, genesis_crystal, primogem, starglitter)
        logging.info("Set resources for user {}: {}".format(self.uid, self.resource))

    def __adjust_fate(self, amount: int):
        self.resource.intertwined_fate = amount
        return True

    def __adjust_crystal(self, amount: int):
        self.resource.genesis_crystal = amount
        return True

    def __adjust_gem(self, amount: int):
        self.resource.primogem = amount
        return True

    def __adjust_star(self, amount: int):
        self.resource.starglitter = amount
        return True

    def __convert_star_to_fate(self, amount: int):
        if self.resource.starglitter < amount * 5:
            return False
        else:
            self.resource.intertwined_fate += amount
            self.resource.starglitter -= amount * 5
            return True

    def __convert_gem_to_fate(self, amount: int):
        if self.resource.primogem < amount * 160:
            return False
        else:
            self.resource.intertwined_fate += amount
            self.resource.primogem -= amount * 160
            return True

    def __convert_crystal_to_gem(self, amount: int):
        if self.resource.genesis_crystal < amount:
            return False
        else:
            self.resource.primogem += amount
            self.resource.genesis_crystal -= amount
            return True

    def update_resource(self, action: consts.GenshinResourceAction, amount: int) -> bool:
        logging.info("Try to update resources for user {}: {}".format(self.uid, self.resource))
        action_dict = {
            consts.GenshinResourceAction.ADJUST_FATE: self.__adjust_fate,
            consts.GenshinResourceAction.ADJUST_CRYSTAL: self.__adjust_crystal,
            consts.GenshinResourceAction.ADJUST_GEM: self.__adjust_gem,
            consts.GenshinResourceAction.ADJUST_STAR: self.__adjust_star,
            consts.GenshinResourceAction.FROM_STAR_TO_FATE: self.__convert_star_to_fate,
            consts.GenshinResourceAction.FROM_GEM_TO_FATE: self.__convert_gem_to_fate,
            consts.GenshinResourceAction.FROM_CRYSTAL_TO_GEM: self.__convert_crystal_to_gem
        }
        selected_action = action_dict[action]
        if selected_action is not None:
            is_success = selected_action(amount)
            if is_success:
                logging.info("Updated resources for user {}: {}".format(self.uid, self.resource))
            else:
                logging.warning(
                    "Cannot updated resources for user {} with GenshinResourceAction{}: {}".format(self.uid, action,
                                                                                                   self.resource))
            return is_success
        else:
            logging.error(
                "Update resources for user {} error: invalid GenshinResourceAction {}".format(self.uid, action))
            return False

    def get_state(self):
        logging.info("Get state for user {}: {}".format(self.uid, self.state))
        return self.state

    def set_state(self, chara_pulls: int, chara_pity: consts.GenshinCharaPityType, weapon_pulls: int,
                  weapon_pity: consts.GenshinWeaponPityType, plan: list[consts.GenshinBannerType]) -> bool:
        logging.info("Try to set state for user {}: {}".format(self.uid, self.state))
        if chara_pity is not consts.GenshinCharaPityType.CHARA_50 and chara_pity is not consts.GenshinCharaPityType.CHARA_100:
            logging.error("Set state for user {} error: invalid GenshinCharaPityType {}".format(self.uid, chara_pity))
            return False
        chara_state = (chara_pulls, chara_pity.value)
        if weapon_pity is consts.GenshinWeaponPityType.WEAPON_50_PATH_0:
            weapon_state = (weapon_pulls, 0, 0)
        elif weapon_pity is consts.GenshinWeaponPityType.WEAPON_100_PATH_0:
            weapon_state = (weapon_pulls, 1, 0)
        elif weapon_pity is consts.GenshinWeaponPityType.WEAPON_50_PATH_1:
            weapon_state = (weapon_pulls, 0, 1)
        elif weapon_pity is consts.GenshinWeaponPityType.WEAPON_100_PATH_1:
            weapon_state = (weapon_pulls, 1, 1)
        elif weapon_pity is consts.GenshinWeaponPityType.WEAPON_50_PATH_2:
            weapon_state = (weapon_pulls, 0, 2)
        elif weapon_pity is consts.GenshinWeaponPityType.WEAPON_100_PATH_2:
            weapon_state = (weapon_pulls, 1, 2)
        else:
            logging.error("Set state for user {} error: invalid GenshinWeaponPityType {}".format(self.uid, weapon_pity))
            return False
        state_plan = []
        for banner in plan:
            if banner is not consts.GenshinBannerType.CHARA and banner is not consts.GenshinBannerType.WEAPON:
                logging.error("Set state for user {} error: invalid GenshinBannerType {}".format(self.uid, banner))
                return False
            state_plan.append(banner.value)
        self.state = model.GenshinWishModelState((chara_state, weapon_state, state_plan))
        logging.info("Set state for user {}: {}".format(self.uid, self.state))
        return True

    def update_state_one_pull(self, banner: consts.GenshinBannerType, pull: consts.GenshinPullResultType) -> bool:
        if banner.value == self.state[2][0]:
            next_states = self.model.get_next_states(self.state)
            update_state = None
            for next_state in next_states:
                if pull == next_state[3]:
                    update_state = next_state[1]
                    break
            if update_state is None:
                return False
            self.state = update_state
            return True
        elif banner.value in self.state[2]:
            index = self.state[2].index(banner.value)

            old_plan = self.state[2][:index]
            new_plan = self.state[2][index:]
            new_state = model.GenshinWishModelState((self.state[0], self.state[1], new_plan))

            next_states = self.model.get_next_states(new_state)
            update_state = None
            for next_state in next_states:
                if pull == next_state[3]:
                    update_state = next_state[1]
                    break
            if update_state is None:
                return False
            self.state = model.GenshinWishModelState((update_state[0], update_state[1], old_plan + update_state[2]))
            return True
        else:
            return True

    def update_state_n_pull(self, banner: consts.GenshinBannerType, pull_list: list[consts.GenshinPullResultType]) -> bool:
        original_state = copy.deepcopy(self.state)
        for result in pull_list:
            is_success = self.update_state_one_pull(banner, result)
            if not is_success:
                self.state = original_state
                return False
        return True

    def trigger_calculator(self):
        pass

    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    user = GenshinUser(1)
    user.update_resource(consts.GenshinResourceAction.ADJUST_STAR, 100)
    print(user.get_resource())
