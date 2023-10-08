import copy
import logging
import os
import pickle
import time

import cupy as cp
import numpy as np
import scipy.sparse as sp

import genshin.wish_model_v2 as model
from calculator.definitions import ROOT_DIR


class WishCalculatorV3:
    def __init__(self, wish_model, init_state, force=False, force_cpu=False):
        self.model = wish_model
        self.init_state = init_state
        self.model_file_path = os.path.join(ROOT_DIR, 'models/genshin_v2_{}.pkl'.format(self.init_state.gen_base64()))

        self.adjacency_list = {}
        self.adjacency_matrix = None
        self.adjacency_matrix_index = {}
        self.result = None

        if force:
            self.__build_model(force_cpu=force_cpu)
            self.__save_cache()
        else:
            if self.__is_cache_exist():
                self.__load_cache()
            else:
                self.__build_model(force_cpu=force_cpu)
                self.__save_cache()

    def __is_cache_exist(self):
        if os.path.exists(self.model_file_path):
            return True
        return False

    def __save_cache(self):
        model_file_path = os.path.join(ROOT_DIR, 'models/genshin_v2_{}.pkl'.format(hash(self.init_state)))
        save = {"index": self.adjacency_matrix_index, "matrix": self.adjacency_matrix, "result": self.result}
        with open(self.model_file_path, 'wb') as f:
            pickle.dump(save, f)

    def __load_cache(self):
        with open(self.model_file_path, 'rb') as f:
            load = pickle.load(f)
        self.adjacency_matrix_index = load["index"]
        self.adjacency_matrix = load["matrix"]
        self.result = load["result"]

    def __build_model(self, force_cpu=False):
        start_time = time.time()
        dfs_set = set()
        dfs_set.add(self.init_state)
        dfs_stack = [self.init_state]
        while len(dfs_stack) > 0:
            curr_state = dfs_stack.pop()
            next_states = self.model.get_next_states(curr_state)
            self.adjacency_list[curr_state] = next_states
            for state in next_states:
                if state[1] not in dfs_set and not state[2]:
                    dfs_set.add(state[1])
                    dfs_stack.append(state[1])
        self.adjacency_matrix_index = {item: index for index, item in enumerate(self.adjacency_list.keys())}
        self.adjacency_matrix = np.zeros((len(self.adjacency_list), len(self.adjacency_list)), dtype=float)
        for curr_state in self.adjacency_list.keys():
            next_states = self.adjacency_list[curr_state]
            for next_state_tuple in next_states:
                self.adjacency_matrix[
                    self.adjacency_matrix_index[curr_state], self.adjacency_matrix_index[next_state_tuple[1]]] = \
                    next_state_tuple[0]

        max_steps = 0
        for banner_type in self.init_state[2]:
            if banner_type == 0:
                max_steps += 180
            elif banner_type == 1:
                max_steps += 231

        target_index = self.adjacency_matrix_index[self.init_state.get_goal_state()[-1]]

        if cp.cuda.runtime.getDeviceCount() == 0 or force_cpu:
            logging.info("Use CPU")
            result = np.zeros((len(self.adjacency_matrix_index), max_steps), dtype=float)
            coo_matrix = sp.coo_matrix(self.adjacency_matrix)
            inter_matrix = None
            first_matrix = copy.deepcopy(coo_matrix)
            for step in range(max_steps):
                logging.debug("Step " + str(step))
                if inter_matrix is None:
                    inter_matrix = coo_matrix
                    target_arrays = inter_matrix.toarray()[:, target_index]
                    result[:, step] = target_arrays
                else:
                    inter_matrix = first_matrix.dot(inter_matrix)
                    target_arrays = inter_matrix.toarray()[:, target_index]
                    result[:, step] = target_arrays
        else:
            logging.info("Use GPU")
            result = cp.zeros((len(self.adjacency_matrix_index), max_steps), dtype=float)
            inter_matrix = cp.sparse.csr_matrix(cp.asarray(self.adjacency_matrix))
            first_matrix = copy.deepcopy(inter_matrix)
            for step in range(max_steps):
                logging.debug("Step " + str(step))
                if step == 0:
                    target_arrays = inter_matrix.toarray()[:, target_index]
                else:
                    inter_matrix = first_matrix.dot(inter_matrix)
                    target_arrays = inter_matrix.toarray()[:, target_index]
                result[:, step] = target_arrays
            result = cp.asnumpy(result)

        self.result = result

        logging.info("Build model in " + str(time.time() - start_time) + " second(s)")

    def get_result(self, start_state):
        if start_state not in self.adjacency_matrix_index:
            return None, False
        else:
            return self.result[self.adjacency_matrix_index[start_state]], True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    g_wish_model = model.GenshinWishModelV2()
    g_state = model.GenshinWishModelState(((0, 0), (0, 0, 0), [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]))
    cal = WishCalculatorV3(g_wish_model, g_state, force=True)
    res = cal.get_result(model.GenshinWishModelState(((0, 0), (0, 0, 0), [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])))
    print(res)
