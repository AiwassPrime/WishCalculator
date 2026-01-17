import copy
import hashlib
import os
import pickle
import time

from loguru import logger

from calculator.definitions import ROOT_DIR
from calculator.endfield import consts

model_file_path = os.path.join(ROOT_DIR, 'models/endfield_chara.pkl')

MAX_PULL = 1200

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

    def get_goal_state(self):
        if len(self[1]) <= 0:
            return [[]]
        
        initial_plan_length = len(self[1])
        
        # Initialize result structure: list of lists, indexed by plan length
        # result[0] = states with plan length = initial_plan_length - 1
        # result[1] = states with plan length = initial_plan_length - 2
        # ...
        # result[initial_plan_length - 1] = states with plan length = 0
        result = [[] for _ in range(initial_plan_length)]
        
        # Use BFS to explore all possible states
        # Use full state tuple as key to avoid hash collision issues
        # Key format: (state[0], tuple(state[1])) to ensure uniqueness
        processed_for_bfs = set()  # States we've already expanded in BFS
        queue = [self]
        
        # Helper function to create a unique key for state comparison
        def get_state_key(state):
            return (state[0], tuple(state[1]))
        
        # Use dict for faster lookup in result lists
        result_sets = [set() for _ in range(initial_plan_length)]
        
        while queue:
            curr_state = queue.pop(0)
            curr_plan_length = len(curr_state[1])
            curr_key = get_state_key(curr_state)
            
            # Skip if already processed (to avoid infinite loops)
            if curr_key in processed_for_bfs:
                continue
            
            # If plan length is 0, add to result and don't continue
            if curr_plan_length == 0:
                result_index = initial_plan_length - 1
                if curr_key not in result_sets[result_index]:
                    result_sets[result_index].add(curr_key)
                    result[result_index].append(curr_state)
                continue
            
            # Mark as processed
            processed_for_bfs.add(curr_key)
            
            # Get next states from model
            model = EndfieldCharaWishModel()
            try:
                next_states = model.get_next_states(curr_state)
            except (KeyError, Exception) as e:
                # If get_next_states fails (e.g., state not in cache), skip
                # This can happen if state exceeds MAX_PULL limit
                continue
            
            # Process each next state
            for _, next_state, _, _ in next_states:
                
                next_plan_length = len(next_state[1])
                next_key = get_state_key(next_state)
                
                # Calculate the index in result list
                # result[0] should contain states with plan length = initial_plan_length - 1
                # result[k] should contain states with plan length = initial_plan_length - 1 - k
                if 0 <= next_plan_length < initial_plan_length:
                    result_index = initial_plan_length - next_plan_length - 1
                    if 0 <= result_index < len(result):
                        # Add to result if not already present (using set for O(1) lookup)
                        if next_key not in result_sets[result_index]:
                            result_sets[result_index].add(next_key)
                            result[result_index].append(next_state)
                
                # Continue BFS if plan length > 0 and state not yet processed
                if next_plan_length > 0 and next_key not in processed_for_bfs:
                    queue.append(next_state)
        
        return result

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


def is_cache_exist():
    if os.path.exists(model_file_path):
        return True
    return False

class EndfieldCharaWishModel:
    chara = []
    chara_cache = {}

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
            logger.info("Build model in " + str(time.time() - start) + " second(s)")

    def __dump_cache(self):
        cache = {
            "chara": self.chara,
            "chara_cache": self.chara_cache
        }
        with open(model_file_path, 'wb') as f:
            pickle.dump(cache, f)

    def __load_cache(self):
        with open(model_file_path, 'rb') as f:
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
        # y: 120/240 count
        # z: 120 is used, 1 means used, 0 means not used
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
                    (1, (0, curr_process[1] + 1, 1), True, consts.EndfieldCharaPullResultType.GET_TARGET))  # get want
                if curr_process[1] + 1 <= MAX_PULL:
                    bfs_queue.append((0, curr_process[1] + 1, 1))
            else:
                if self.chara[curr_process[0] + 1] >= 1:
                    # 1/2 chance to get want
                    chara_cache.setdefault(curr_process, []).append(
                        (0.5, (0, curr_process[1] + 1, 1), True, consts.EndfieldCharaPullResultType.GET_TARGET))  # get want
                    chara_cache.setdefault(curr_process, []).append(
                        (0.5, (0, curr_process[1] + 1, curr_process[2]), False, consts.EndfieldCharaPullResultType.GET_PITY))  # get pity
                    if curr_process[1] + 1 <= MAX_PULL:
                        bfs_queue.append((0, curr_process[1] + 1, 1))
                        bfs_queue.append((0, curr_process[1] + 1, curr_process[2]))
                else:
                    # 0.8% + (n - 65) * 5% chance to get 6 star, 1/2 chance to get want
                    chara_cache.setdefault(curr_process, []).append(
                        (self.chara[curr_process[0] + 1] * 0.5, (0, curr_process[1] + 1, 1), True,
                         consts.EndfieldCharaPullResultType.GET_TARGET))  # get want
                    chara_cache.setdefault(curr_process, []).append(
                        (self.chara[curr_process[0] + 1] * 0.5, (0, curr_process[1] + 1, curr_process[2]), False,
                         consts.EndfieldCharaPullResultType.GET_PITY))  # get pity
                    chara_cache.setdefault(curr_process, []).append(
                        (1 - self.chara[curr_process[0] + 1], (curr_process[0] + 1, curr_process[1] + 1, curr_process[2]), False,
                         consts.EndfieldCharaPullResultType.GET_NOTHING))  # get nothing
                    if curr_process[1] + 1 <= MAX_PULL:
                        bfs_queue.append((0, curr_process[1] + 1, 1))
                        bfs_queue.append((0, curr_process[1] + 1, curr_process[2]))
                        bfs_queue.append((curr_process[0] + 1, curr_process[1] + 1, curr_process[2]))
        self.chara_cache = chara_cache

    def get_next_states(self, curr_state: EndfieldCharaWishModelState) -> list[
        tuple[float, EndfieldCharaWishModelState, bool, consts.EndfieldCharaPullResultType]]:
        if len(curr_state[1]) <= 0:
            return [(1.0, curr_state, True, consts.EndfieldCharaPullResultType.TERMINAL_STATE)]
        if len(self.chara_cache[curr_state[0]]) <= 0:
            raise Exception("Unexpected state: " + str(curr_state))

        res = []
        for chance, chara_state, is_want, result in self.chara_cache[curr_state[0]]:
            if chara_state[1] % 240 == 0:
                if is_want and len(curr_state[1]) > 1:
                    res.append((chance, EndfieldCharaWishModelState((chara_state, curr_state[1][2:])), False, result))
                else:
                    res.append((chance, EndfieldCharaWishModelState((chara_state, curr_state[1][1:])), False, result))
            else:
                if is_want:
                    res.append((chance, EndfieldCharaWishModelState((chara_state, curr_state[1][1:])), False, result))
                else:
                    res.append((chance, EndfieldCharaWishModelState((chara_state, curr_state[1])), False, result))
        return res


if __name__ == "__main__":
    module = EndfieldCharaWishModel(force=True)
    state = EndfieldCharaWishModelState(((0, 0, 0), [0]))
    n = module.get_next_states(state)
    goal_states = state.get_goal_state()
    print(f"Goal states: {len(goal_states)} groups")
    for i, group in enumerate(goal_states):
        print(f"Group {i} (plan length {len(state[1]) - 1 - i}): {len(group)} states")
