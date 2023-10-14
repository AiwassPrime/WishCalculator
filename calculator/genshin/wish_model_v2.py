import base64
import copy
import hashlib
import logging
import pickle
import os
import time

from calculator.genshin import consts
from calculator.definitions import ROOT_DIR

model_file_path = os.path.join(ROOT_DIR, 'models/genshin_v2.pkl')


class GenshinWishModelState(tuple[tuple[int, int], tuple[int, int, int], list[int]]):
    def __str__(self):
        return str((self[0], self[1], ''.join(map(str, self[2]))))

    def __hash__(self):
        return hash((self[0], self[1], len(self[2])))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def gen_base64(self):
        hash_object = hashlib.sha256(str(self).encode())
        hash_value = hash_object.hexdigest()
        return hash_value

    def get_reduced_state(self):
        if len(self[2]) <= 0:
            return []
        result_list = []
        for i in range(1, len(self[2]) + 1):
            result_list.append(GenshinWishModelState((self[0], self[1], self[2][:i])))
        return result_list

    def get_goal_state(self):
        if len(self[2]) <= 0:
            return []
        result_list = []
        curr_target = self[2][0]
        plan = self[2][1:]
        curr_state = copy.deepcopy(self)
        while len(plan) >= 0:
            if curr_target == 0:
                curr_state = GenshinWishModelState(((0, 0), curr_state[1], plan))
                result_list.append(curr_state)
            else:
                curr_state = GenshinWishModelState((curr_state[0], (0, 0, 0), plan))
                result_list.append(curr_state)
            if len(plan) == 0:
                break
            curr_target = plan[0]
            plan = plan[1:]

        return result_list


def is_cache_exist():
    if os.path.exists(model_file_path):
        return True
    return False


class GenshinWishModelV2:
    chara = []
    weapon = []
    chara_cache = {}
    weapon_cache = {}

    def __init__(self, force=False):
        if not force and len(self.chara) != 0:
            return
        if not force and is_cache_exist():
            self.__load_cache()
        else:
            start = time.time()
            self.__fill_chance()
            self.__fill_cache()
            self.__dump_cache()
            logging.info("Build model in " + str(time.time() - start) + " second(s)")

    def __dump_cache(self):
        cache = {
            "chara": self.chara,
            "weapon": self.weapon,
            "chara_cache": self.chara_cache,
            "weapon_cache": self.weapon_cache
        }
        with open(model_file_path, 'wb') as f:
            pickle.dump(cache, f)

    def __load_cache(self):
        with open(model_file_path, 'rb') as f:
            cache = pickle.load(f)
            self.chara = cache["chara"]
            self.weapon = cache["weapon"]
            self.chara_cache = cache["chara_cache"]
            self.weapon_cache = cache["weapon_cache"]

    def __fill_chance(self):
        chara_list = []
        for i in range(100):
            if i <= 73:
                chara_list.append(60 / 10000)
            else:
                chara_list.append((60 + (600 * (i - 73))) / 10000)
        self.chara = chara_list
        weapon_list = []
        for i in range(100):
            if i <= 62:
                weapon_list.append(70 / 10000)
            else:
                weapon_list.append((70 + (700 * (i - 62))) / 10000)
        self.weapon = weapon_list

    def __fill_cache(self):
        chara_cache = {}
        bfs_queue = [(0, 0)]
        bfs_set = set()
        while len(bfs_queue) > 0:
            curr_process = bfs_queue.pop()
            if self.chara[curr_process[0] + 1] >= 1:
                if curr_process[1] >= 1:
                    chara_cache.setdefault(curr_process, []).append(
                        (1.0, (0, 0), True, consts.GenshinPullResultType.GET_TARGET))  # get want
                else:
                    chara_cache.setdefault(curr_process, []).append(
                        (0.5, (0, 0), True, consts.GenshinPullResultType.GET_TARGET))  # get want
                    chara_cache.setdefault(curr_process, []).append(
                        (0.5, (0, 1), False, consts.GenshinPullResultType.GET_PITY))  # get pity
                    if (0, 1) not in bfs_set:
                        bfs_set.add((0, 1))
                        bfs_queue.append((0, 1))
            else:
                if curr_process[1] >= 1:
                    chara_cache.setdefault(curr_process, []).append(
                        (self.chara[curr_process[0] + 1], (0, 0), True,
                         consts.GenshinPullResultType.GET_TARGET))  # get want
                    chara_cache.setdefault(curr_process, []).append((
                        1.0 - self.chara[curr_process[0] + 1], (curr_process[0] + 1, 1), False,
                        consts.GenshinPullResultType.GET_NOTHING))  # get nothing
                    if (curr_process[0] + 1, 1) not in bfs_set:
                        bfs_set.add((curr_process[0] + 1, 1))
                        bfs_queue.append((curr_process[0] + 1, 1))
                else:
                    chara_cache.setdefault(curr_process, []).append(
                        (self.chara[curr_process[0] + 1] * 0.5, (0, 0), True,
                         consts.GenshinPullResultType.GET_TARGET))  # get want
                    chara_cache.setdefault(curr_process, []).append(
                        (self.chara[curr_process[0] + 1] * 0.5, (0, 1), False,
                         consts.GenshinPullResultType.GET_PITY))  # get pity
                    chara_cache.setdefault(curr_process, []).append((
                        1.0 - self.chara[curr_process[0] + 1], (curr_process[0] + 1, 0), False,
                        consts.GenshinPullResultType.GET_NOTHING))  # get nothing
                    if (0, 1) not in bfs_set:
                        bfs_set.add((0, 1))
                        bfs_queue.append((0, 1))
                    if (curr_process[0] + 1, 0) not in bfs_set:
                        bfs_set.add((curr_process[0] + 1, 0))
                        bfs_queue.append((curr_process[0] + 1, 0))
        self.chara_cache = chara_cache

        weapon_cache = {}
        bfs_queue = [(0, 0, 0), (0, 1, 0)]
        bfs_set = set()
        while len(bfs_queue) > 0:
            curr_process = bfs_queue.pop()
            if self.weapon[curr_process[0] + 1] >= 1:
                if curr_process[2] >= 2:
                    weapon_cache.setdefault(curr_process, []).append(
                        (1.0, (0, 0, 0), True, consts.GenshinPullResultType.GET_TARGET))  # get want
                elif curr_process[1] >= 1:
                    weapon_cache.setdefault(curr_process, []).append(
                        (0.5, (0, 0, 0), True, consts.GenshinPullResultType.GET_TARGET))  # get want
                    weapon_cache.setdefault(curr_process, []).append(
                        (0.5, (0, 0, curr_process[2] + 1), False,
                         consts.GenshinPullResultType.GET_NON_TARGET))  # get other
                    if (0, 0, curr_process[2] + 1) not in bfs_set:
                        bfs_set.add((0, 0, curr_process[2] + 1))
                        bfs_queue.append((0, 0, curr_process[2] + 1))
                else:
                    weapon_cache.setdefault(curr_process, []).append(
                        (0.375, (0, 0, 0), True, consts.GenshinPullResultType.GET_TARGET))  # get want
                    weapon_cache.setdefault(curr_process, []).append(
                        (0.375, (0, 0, curr_process[2] + 1), False,
                         consts.GenshinPullResultType.GET_NON_TARGET))  # get other
                    weapon_cache.setdefault(curr_process, []).append(
                        (0.25, (0, 1, curr_process[2] + 1), False, consts.GenshinPullResultType.GET_PITY))  # get pity
                    if (0, 0, curr_process[2] + 1) not in bfs_set:
                        bfs_set.add((0, 0, curr_process[2] + 1))
                        bfs_queue.append((0, 0, curr_process[2] + 1))
                    if (0, 1, curr_process[2] + 1) not in bfs_set:
                        bfs_set.add((0, 1, curr_process[2] + 1))
                        bfs_queue.append((0, 1, curr_process[2] + 1))
            else:
                if curr_process[2] >= 2:
                    weapon_cache.setdefault(curr_process, []).append(
                        (self.weapon[curr_process[0] + 1], (0, 0, 0), True,
                         consts.GenshinPullResultType.GET_TARGET))  # get want
                    weapon_cache.setdefault(curr_process, []).append((
                        1 - self.weapon[curr_process[0] + 1], (curr_process[0] + 1, curr_process[1], 2),
                        False, consts.GenshinPullResultType.GET_NOTHING))  # get nothing
                    if (curr_process[0] + 1, curr_process[1], 2) not in bfs_set:
                        bfs_set.add((curr_process[0] + 1, curr_process[1], 2))
                        bfs_queue.append((curr_process[0] + 1, curr_process[1], 2))
                elif curr_process[1] >= 1:
                    weapon_cache.setdefault(curr_process, []).append(
                        (self.weapon[curr_process[0] + 1] * 0.5, (0, 0, 0), True,
                         consts.GenshinPullResultType.GET_TARGET))  # get want
                    weapon_cache.setdefault(curr_process, []).append((
                        self.weapon[curr_process[0] + 1] * 0.5, (0, 0, curr_process[2] + 1), False,
                        consts.GenshinPullResultType.GET_NON_TARGET))  # get other
                    weapon_cache.setdefault(curr_process, []).append((
                        1.0 - self.weapon[curr_process[0] + 1], (curr_process[0] + 1, 1, curr_process[2]),
                        False, consts.GenshinPullResultType.GET_NOTHING))  # get nothing
                    if (0, 0, curr_process[2] + 1) not in bfs_set:
                        bfs_set.add((0, 0, curr_process[2] + 1))
                        bfs_queue.append((0, 0, curr_process[2] + 1))
                    if (curr_process[0] + 1, 1, curr_process[2]) not in bfs_set:
                        bfs_set.add((curr_process[0] + 1, 1, curr_process[2]))
                        bfs_queue.append((curr_process[0] + 1, 1, curr_process[2]))
                else:
                    weapon_cache.setdefault(curr_process, []).append(
                        (self.weapon[curr_process[0] + 1] * 0.375, (0, 0, 0), True,
                         consts.GenshinPullResultType.GET_TARGET))  # get want
                    weapon_cache.setdefault(curr_process, []).append((
                        self.weapon[curr_process[0] + 1] * 0.375, (0, 0, curr_process[2] + 1), False,
                        consts.GenshinPullResultType.GET_NON_TARGET))  # get other
                    weapon_cache.setdefault(curr_process, []).append((
                        self.weapon[curr_process[0] + 1] * 0.25, (0, 1, curr_process[2] + 1), False,
                        consts.GenshinPullResultType.GET_PITY))  # get pity
                    weapon_cache.setdefault(curr_process, []).append((
                        1.0 - self.weapon[curr_process[0] + 1], (curr_process[0] + 1, 0, curr_process[2]),
                        False, consts.GenshinPullResultType.GET_NOTHING))  # get nothing
                    if (0, 0, curr_process[2] + 1) not in bfs_set:
                        bfs_set.add((0, 0, curr_process[2] + 1))
                        bfs_queue.append((0, 0, curr_process[2] + 1))
                    if (0, 1, curr_process[2] + 1) not in bfs_set:
                        bfs_set.add((0, 1, curr_process[2] + 1))
                        bfs_queue.append((0, 1, curr_process[2] + 1))
                    if (curr_process[0] + 1, 0, curr_process[2]) not in bfs_set:
                        bfs_set.add((curr_process[0] + 1, 0, curr_process[2]))
                        bfs_queue.append((curr_process[0] + 1, 0, curr_process[2]))
        self.weapon_cache = weapon_cache

    def get_next_states(self, curr_state: GenshinWishModelState) -> list[tuple[float, GenshinWishModelState, bool, consts.GenshinPullResultType]]:
        if len(curr_state[2]) <= 0:
            return [(1.0, curr_state, True, consts.GenshinPullResultType.TERMINAL_STATE)]
        if curr_state[2][0] == 0:
            if len(self.chara_cache[curr_state[0]]) <= 0:
                raise Exception("Unexpected state: " + str(curr_state))
            return [(chance, GenshinWishModelState(
                (chara_state, curr_state[1], (curr_state[2][1:] if is_want else curr_state[2]))), False, result) for
                    chance, chara_state, is_want, result in self.chara_cache[curr_state[0]]]
        else:
            if len(self.weapon_cache[curr_state[1]]) <= 0:
                raise Exception("Unexpected state: " + str(curr_state))
            return [(chance, GenshinWishModelState(
                (curr_state[0], weapon_state, (curr_state[2][1:] if is_want else curr_state[2]))), False, result) for
                    chance, weapon_state, is_want, result in self.weapon_cache[curr_state[1]]]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = GenshinWishModelV2(force=True)
    res = model.get_next_states(GenshinWishModelState(((74, 0), (1, 0, 0), [0, 0, 0, 0, 0, 0, 1])))
    print(res)
