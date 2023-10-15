import copy
import logging
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt

import wish_model_v2 as model
from calculator.genshin import consts
import matplotlib.colors as mcolors
from calculator.wish_calculator_v3 import WishCalculatorV3


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
        if self.resource.intertwined_fate > 0:
            self.resource.intertwined_fate -= 1
        elif 0 < self.resource.primogem < 160:
            if self.resource.genesis_crystal >= (160 - self.resource.primogem):
                self.resource.genesis_crystal -= (160 - self.resource.primogem)
                self.resource.primogem = 0
            else:
                return False
        elif self.resource.primogem >= 160:
            self.resource.primogem -= 160
        elif self.resource.genesis_crystal >= 160:
            self.resource.genesis_crystal -= 160
        else:
            return False

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

    def update_state_n_pull(self, banner: consts.GenshinBannerType,
                            pull_list: list[consts.GenshinPullResultType]) -> bool:
        original_state = copy.deepcopy(self.state)
        original_resource = copy.deepcopy(self.resource)
        for result in pull_list:
            is_success = self.update_state_one_pull(banner, result)
            if not is_success:
                self.state = original_state
                self.resource = original_resource
                return False
        return True

    def trigger_calculator(self):
        state_list = self.state.get_reduced_state()
        for state in state_list:
            self.calculator[state] = WishCalculatorV3(self.model, state)

    def get_total_pull(self):
        return self.resource.intertwined_fate + self.resource.primogem // 160 + self.resource.genesis_crystal // 160 \
            + self.resource.starglitter // 5

    def get_raw_result(self):
        no_regenerate = True
        goal_states = self.state.get_reduced_state()[::-1]
        result_list = {}
        keys_list = list(self.calculator.keys())[::-1]
        for index, key in enumerate(keys_list, start=0):
            if index >= len(goal_states):
                break
            result, have_result = self.calculator[key].get_result(goal_states[index])
            if have_result:
                no_regenerate = False
                result_list[goal_states[index]] = result
        return result_list, no_regenerate


def process_result(result):
    row = 0
    column = 0
    for state, res in result.items():
        row += 1
        if len(res) > column:
            column = len(res)
    grid = [[1 for _ in range(column)] for _ in range(row)]
    for index_x, stats in enumerate(result.values()):
        for index_y, stat in enumerate(stats):
            grid[index_x][index_y] = stat
    return result, grid, (row, column)


def process_result_agg(result):
    agg_result = {}
    row = 0
    column = 0
    for state, res in result.items():
        row += 1
        if len(res) > column:
            column = len(res)
        index_10 = np.searchsorted(res, "0.1", side='right')
        index_25 = np.searchsorted(res, "0.25", side='right')
        index_50 = np.searchsorted(res, "0.5", side='right')
        index_75 = np.searchsorted(res, "0.75", side='right')
        index_90 = np.searchsorted(res, "0.9", side='right')
        agg_result[state] = [index_10, index_25, index_50, index_75, index_90]
    grid = [[0 for _ in range(column)] for _ in range(row)]
    for index, agg_stats in enumerate(agg_result.values()):
        for stats in agg_stats:
            grid[index][stats] = 1
    return agg_result, grid, (row, column)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    user = GenshinUser(1)

    user.update_resource(consts.GenshinResourceAction.ADJUST_FATE, 47)
    user.update_resource(consts.GenshinResourceAction.ADJUST_GEM, 62216)
    user.update_resource(consts.GenshinResourceAction.ADJUST_CRYSTAL, 38588)
    user.update_resource(consts.GenshinResourceAction.ADJUST_STAR, 78)
    pull = user.get_total_pull()

    user.set_state(8, consts.GenshinCharaPityType.CHARA_100, 1, consts.GenshinWeaponPityType.WEAPON_50_PATH_0,
                   [consts.GenshinBannerType.CHARA, consts.GenshinBannerType.CHARA, consts.GenshinBannerType.CHARA,
                    consts.GenshinBannerType.CHARA, consts.GenshinBannerType.CHARA, consts.GenshinBannerType.CHARA,
                    consts.GenshinBannerType.CHARA, consts.GenshinBannerType.WEAPON])

    user.trigger_calculator()

    raw, is_success = user.get_raw_result()
    _, graph, dem = process_result(raw)
    agg, _, _ = process_result_agg(raw)

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=1)

    aspect_ratio = dem[0] / dem[1]
    plt.figure(figsize=(aspect_ratio, 1))
    plt.imshow(graph, cmap=cmap, interpolation='none', aspect='auto', norm=norm)

    plt.fill_betweenx(y=[-0.5, 7.5], x1=pull, x2=pull, color='blue')
    for index, agg_stats in enumerate(agg.values()):
        for stats in agg_stats:
            plt.fill_betweenx(y=[index-0.5, index+0.5], x1=stats, x2=stats, color='black')

    cbar = plt.colorbar()
    cbar.set_label('Values')

    x_indices = np.arange(0, dem[1], 50)
    x_labels = [str(index) for index in x_indices]
    y_indices = np.arange(0, dem[0], 1)
    y_labels = [agg_s.get_plan_str() for agg_s in agg.keys()]
    plt.xticks(ticks=x_indices, labels=x_labels)
    plt.yticks(ticks=y_indices, labels=y_labels)

    plt.text(pull, -0.8, 'You={}'.format(pull), color='blue', fontsize=12, ha='center', va='center')

    plt.show()
