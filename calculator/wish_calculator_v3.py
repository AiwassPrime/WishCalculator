import time

import numpy as np
import scipy.sparse as sp

import genshin.wish_model_v2 as model


class WishCalculatorV3:
    def __init__(self, wish_model, init_state):
        self.model = wish_model
        self.init_state = init_state

        self.adjacency_list = {}
        self.adjacency_matrix = None
        self.adjacency_matrix_index = {}

        self.__build_model()

    def __build_model(self):
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
            self.adjacency_matrix[
                self.adjacency_matrix_index[curr_state], [self.adjacency_matrix_index[state[1]] for state in
                                                          next_states]] = [state[0] for state in next_states]
        inter_matrix = None
        first_matrix = self.adjacency_matrix

        combined_array = None
        target_index = self.adjacency_matrix_index[model.GenshinWishModelState(((0, 0), (0, 0, 0), []))]
        max_steps = 0
        for banner_type in [0, 0, 0, 0, 0, 0, 0, 1]:
            if banner_type == 0:
                max_steps += 180
            elif banner_type == 1:
                max_steps += 231
        for step in range(max_steps):
            if inter_matrix is None:
                coo_matrix = sp.coo_matrix(self.adjacency_matrix)
                inter_matrix = coo_matrix
                first_matrix = coo_matrix
                combined_array = coo_matrix.toarray()[:, target_index]
            else:
                inter_matrix = first_matrix.dot(inter_matrix)
                combined_array = np.vstack((combined_array, inter_matrix.toarray()[:, target_index]))
        print("Build adjacency matrix in " + str(time.time() - start_time) + " second(s)")


if __name__ == "__main__":
    g_wish_model = model.GenshinWishModelV2()
    g_state = model.GenshinWishModelState(((0, 0), (0, 0, 0), [0, 0, 0, 0, 0, 0, 0, 1]))
    cal = WishCalculatorV3(g_wish_model, g_state)
    print(len(cal.adjacency_list))
