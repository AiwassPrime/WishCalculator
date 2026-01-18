import copy
import hashlib
import os
import pickle
import time

from loguru import logger

from calculator.definitions import ROOT_DIR
from calculator.endfield import consts

class EndfieldCharaWishModelState(tuple[tuple[int, int, int], list[int]]):
    def __new__(cls, state_tuple):
        if state_tuple[0][1] >= 120 and state_tuple[0][2] != 1:
            raise Exception("State is invalid")
        return super().__new__(cls, state_tuple)

    def __str__(self):
        return str((self[0], ''.join(map(str, self[1]))))

    def __hash__(self):
        return hash((self[0], len(self[1])))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def gen_base64(self):
        hash_object = hashlib.sha256(str(self).encode())
        hash_value = hash_object.hexdigest()
        return hash_value

    def get_reduced_state(self):
        if len(self[1]) <= 0:
            return []
        result_list = []
        for i in range(1, len(self[1]) + 1):
            result_list.append(EndfieldCharaWishModelState((self[0], self[1][:i])))
        return result_list

    def get_plan_str(self):
        plan = self[1]
        if len(plan) == 0:
            return ""
        plan_0 = -1
        plan_1 = 0
        result_plan = []
        curr_type = plan[0]
        pull_type = plan[0]
        for pull_type in plan:
            if pull_type == 0:
                plan_0 += 1
                if curr_type != pull_type:
                    result_plan.append("R{}".format(plan_1))
                    curr_type = pull_type
            else:
                raise Exception("Unknown banner type")
        if pull_type == 0:
            result_plan.append("O{}".format(plan_0))
        else:
            raise Exception("Unknown banner type")
        return "-".join(result_plan)


class EndfieldCharaWishModel:
    model_name = 'endfield_chara'
    model_file_path = ''
    chara = []
    chara_cache = {}
    have_30_extra = True

    def __init__(self, force=False, have_30_extra=True):
        self.have_30_extra = have_30_extra
        if have_30_extra:
            self.model_name += '_30_extra'
        self.model_file_path = os.path.join(ROOT_DIR, 'models/' + self.model_name + '.pkl')
        if not force and len(self.chara) != 0:
            return
        if not force and self.__is_cache_exist():
            self.__load_cache()
        else:
            start = time.time()
            self.__fill_chance()
            self.__fill_cache()
            self.__dump_cache()
            logger.info("Build model in " + str(time.time() - start) + " second(s)")

    def __is_cache_exist(self):
        if os.path.exists(self.model_file_path):
            return True
        return False

    def __dump_cache(self):
        cache = {
            "chara": self.chara,
            "chara_cache": self.chara_cache
        }
        with open(self.model_file_path, 'wb') as f:
            pickle.dump(cache, f)

    def __load_cache(self):
        with open(self.model_file_path, 'rb') as f:
            cache = pickle.load(f)
            self.chara = cache["chara"]
            self.chara_cache = cache["chara_cache"]

    def __fill_chance(self):
        chara_list = []
        for i in range(90):
            if i <= 65:
                chara_list.append(80 / 10000)
            elif i >= 80:
                chara_list.append(1)
            else:
                chara_list.append((80 + (500 * (i - 65))) / 10000)
        self.chara = chara_list

    def __fill_cache(self):
        chara_cache = {}
        # tuple (x, y)
        # x: 80 count
        # y: 30/120/240 count
        # z: one time bonus for 120, 1 means used, 0 means not used
        bfs_queue = [(x, 0, 0) for x in range(80)]
        bfs_queue.extend([(x, 0, 1) for x in range(80)])
        bfs_set = set()
        while len(bfs_queue) > 0:
            curr_process = bfs_queue.pop()
            if curr_process in bfs_set:
                continue
            bfs_set.add(curr_process)
            if curr_process[2] == 0 and curr_process[1] + 1 == 120:
                # just hit 120, must get want
                chara_cache.setdefault(curr_process, []).append(
                    (1, (0, curr_process[1] + 1, 1), 1, consts.EndfieldCharaPullResultType.GET_TARGET))  # get want
                bfs_queue.append((0, curr_process[1] + 1, 1))
            else:
                if self.chara[curr_process[0] + 1] >= 1:
                    if curr_process[1] + 1 < 240:
                        if curr_process[2] != 0 and curr_process[1] == 0:
                            # 1/2 chance to get want
                            chara_cache.setdefault(curr_process, []).append(
                                (0.5, (0, curr_process[1] + 1, 1), 2, consts.EndfieldCharaPullResultType.GET_TARGET))  # get want, also get one more
                            chara_cache.setdefault(curr_process, []).append(
                                (0.5, (0, curr_process[1] + 1, curr_process[2]), 1, consts.EndfieldCharaPullResultType.GET_PITY))  # get pity, also get one more
                            bfs_queue.append((0, curr_process[1] + 1, 1))
                            bfs_queue.append((0, curr_process[1] + 1, curr_process[2]))
                        else:
                            # 1/2 chance to get want
                            chara_cache.setdefault(curr_process, []).append(
                                (0.5, (0, curr_process[1] + 1, 1), 1, consts.EndfieldCharaPullResultType.GET_TARGET))  # get want
                            chara_cache.setdefault(curr_process, []).append(
                                (0.5, (0, curr_process[1] + 1, curr_process[2]), 0, consts.EndfieldCharaPullResultType.GET_PITY))  # get pity
                            bfs_queue.append((0, curr_process[1] + 1, 1))
                            bfs_queue.append((0, curr_process[1] + 1, curr_process[2]))
                    else:
                        # next is 240, reset to 0
                        chara_cache.setdefault(curr_process, []).append(
                            (0.5, (0, 0, 1), 1, consts.EndfieldCharaPullResultType.GET_TARGET))  # get want
                        chara_cache.setdefault(curr_process, []).append(
                            (0.5, (0, 0, curr_process[2]), 0, consts.EndfieldCharaPullResultType.GET_PITY))  # get pity
                        bfs_queue.append((0, 0, 1))
                        bfs_queue.append((0, 0, curr_process[2]))
                else:
                    if curr_process[1] + 1 < 240:
                        if curr_process[2] != 0 and curr_process[1] == 0:
                            # 0.8% + (n - 65) * 5% chance to get 6 star, 1/2 chance to get want
                            chara_cache.setdefault(curr_process, []).append(
                                (self.chara[curr_process[0] + 1] * 0.5, (0, curr_process[1] + 1, 1), 2,
                                 consts.EndfieldCharaPullResultType.GET_TARGET))  # get want, also get one more
                            chara_cache.setdefault(curr_process, []).append(
                                (self.chara[curr_process[0] + 1] * 0.5, (0, curr_process[1] + 1, curr_process[2]), 1,
                                 consts.EndfieldCharaPullResultType.GET_PITY))  # get pity, also get one more
                            chara_cache.setdefault(curr_process, []).append(
                                (1 - self.chara[curr_process[0] + 1], (curr_process[0] + 1, curr_process[1] + 1, curr_process[2]), 1,
                                 consts.EndfieldCharaPullResultType.GET_NOTHING))  # get nothing, also get one more
                            bfs_queue.append((0, curr_process[1] + 1, 1))
                            bfs_queue.append((0, curr_process[1] + 1, curr_process[2]))
                            bfs_queue.append((curr_process[0] + 1, curr_process[1] + 1, curr_process[2]))
                        else:
                            # 0.8% + (n - 65) * 5% chance to get 6 star, 1/2 chance to get want
                            chara_cache.setdefault(curr_process, []).append(
                                (self.chara[curr_process[0] + 1] * 0.5, (0, curr_process[1] + 1, 1), 1,
                                 consts.EndfieldCharaPullResultType.GET_TARGET))  # get want
                            chara_cache.setdefault(curr_process, []).append(
                                (self.chara[curr_process[0] + 1] * 0.5, (0, curr_process[1] + 1, curr_process[2]), 0,
                                 consts.EndfieldCharaPullResultType.GET_PITY))  # get pity
                            chara_cache.setdefault(curr_process, []).append(
                                (1 - self.chara[curr_process[0] + 1], (curr_process[0] + 1, curr_process[1] + 1, curr_process[2]), 0,
                                 consts.EndfieldCharaPullResultType.GET_NOTHING))  # get nothing
                            bfs_queue.append((0, curr_process[1] + 1, 1))
                            bfs_queue.append((0, curr_process[1] + 1, curr_process[2]))
                            bfs_queue.append((curr_process[0] + 1, curr_process[1] + 1, curr_process[2]))
                    else:
                        # next is 240, reset to 0
                        chara_cache.setdefault(curr_process, []).append(
                            (self.chara[curr_process[0] + 1] * 0.5, (0, 0, 1), 1,
                             consts.EndfieldCharaPullResultType.GET_TARGET))  # get want
                        chara_cache.setdefault(curr_process, []).append(
                            (self.chara[curr_process[0] + 1] * 0.5, (0, 0, curr_process[2]), 0,
                             consts.EndfieldCharaPullResultType.GET_PITY))  # get pity
                        chara_cache.setdefault(curr_process, []).append(
                            (1 - self.chara[curr_process[0] + 1], (curr_process[0] + 1, 0, curr_process[2]), 0,
                             consts.EndfieldCharaPullResultType.GET_NOTHING))  # get nothing
                        bfs_queue.append((0, 0, 1))
                        bfs_queue.append((0, 0, curr_process[2]))
                        bfs_queue.append((curr_process[0] + 1, 0, curr_process[2]))
        self.chara_cache = chara_cache

    def get_next_states(self, curr_state: EndfieldCharaWishModelState) -> list[
        tuple[float, EndfieldCharaWishModelState, bool, consts.EndfieldCharaPullResultType]]:
        if len(curr_state[1]) <= 0:
            return [(1.0, curr_state, True, consts.EndfieldCharaPullResultType.TERMINAL_STATE)]
        if len(self.chara_cache[curr_state[0]]) <= 0:
            raise Exception("Unexpected state: " + str(curr_state))

        res = []
        for chance, chara_state, num_of_charas, result in self.chara_cache[curr_state[0]]:
            res.append((chance, EndfieldCharaWishModelState((chara_state, curr_state[1][num_of_charas:])), False, result))
        return res

    def get_goal_state(self, curr_state: EndfieldCharaWishModelState):
        if len(curr_state[1]) <= 0:
            return [[]]
        result = [[] for _ in range(len(curr_state[1]))]
        bfs_queue = [curr_state]
        bfs_set = set()
        add_set = set()
        while len(bfs_queue) > 0:
            curr_process = bfs_queue.pop()
            if curr_process in bfs_set:
                continue
            bfs_set.add(curr_process)
            for _, next_state, _, _ in self.get_next_states(curr_process):
                if len(next_state[1]) >= len(curr_state[1]):
                    bfs_queue.append(next_state)
                elif len(next_state[1]) <= 0:
                    if next_state not in add_set:
                        add_set.add(next_state)
                        result[-1].append(next_state)
                else:
                    if next_state not in add_set:
                        add_set.add(next_state)
                        result[len(curr_state[1])-len(next_state[1])-1].append(next_state)
                    bfs_queue.append(next_state)
        return result


if __name__ == "__main__":
    module = EndfieldCharaWishModel(force=True, have_30_extra=False)
    state = EndfieldCharaWishModelState(((60, 1, 1), [0, 0, 0]))
    g = module.get_next_states(state)
    print(g)
