import copy
import os
import pickle
import time

import numpy as np
from matplotlib import pyplot
from pydtmc import MarkovChain, plot_sequence
import scipy.sparse as sp

import genshin.wish_model as model
import concurrent.futures
import threading


class WishCalculatorV2:
    def __init__(self, wish_model, force_calculate=False):
        self.plan = wish_model.get_plan()
        self.wish_model = wish_model

        self.mc = {key: None for key in range(len(self.plan))}
        self.mc_metrix = {key: None for key in range(len(self.plan))}
        self.mc_index_dict = {key: {} for key in range(len(self.plan))}
        self.mc_index_list = {key: [] for key in range(len(self.plan))}
        self.states_list = {key: {} for key in range(len(self.plan))}

        self.result = None

        self.curr_wish_model = None
        self.curr_plan = None
        self.bfs_set = set()
        self.accumulator_lock = threading.Lock()

        model_file_path = './models/' + self.__gen_model_file_name()
        dict_file_path = './models/' + self.__gen_dict_file_name()
        result_file_path = './models/' + self.__gen_result_file_name()
        if force_calculate:
            self._build_calculator()
        elif os.path.exists(model_file_path) and os.path.exists(dict_file_path) and os.path.exists(result_file_path):
            with open(model_file_path, 'rb') as f:
                self.mc = pickle.load(f)
            with open(dict_file_path, 'rb') as f:
                self.mc_index_dict = pickle.load(f)
            with open(result_file_path, 'rb') as f:
                self.result = pickle.load(f)
        else:
            self._build_calculator()

    def __gen_model_file_name(self):
        return self.wish_model.get_model_name() + '_model.pkl'

    def __gen_dict_file_name(self):
        return self.wish_model.get_model_name() + '_dict.pkl'

    def __gen_result_file_name(self):
        return self.wish_model.get_model_name() + '_result.pkl'

    def __process_model(self, curr_state):
        next_states = curr_state.get_next_states()
        with self.accumulator_lock:
            if len(next_states) == 1 and next_states[0][1] == curr_state:
                self.states_list[len(self.curr_plan) - 1][curr_state] = [next_states[0]]
            else:
                if curr_state not in self.states_list[len(self.curr_plan) - 1].keys():
                    self.states_list[len(self.curr_plan) - 1][curr_state] = next_states
                for item in next_states:
                    self.bfs_set.add(item[1])

    def __build_adj_list(self):
        self.curr_wish_model = copy.deepcopy(self.wish_model)
        self.curr_plan = self.curr_wish_model.get_plan()
        while len(self.curr_plan) > 0:
            states = {self.curr_wish_model}
            while len(states) > 0:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    executor.map(self.__process_model, states)
                states = copy.deepcopy(self.bfs_set)
                self.bfs_set.clear()
            self.curr_wish_model = self.curr_wish_model.del_last_plan()
            self.curr_plan = self.curr_wish_model.get_plan()

    def __build_model(self):
        for i in range(len(self.plan)):
            self.mc_metrix[i] = np.zeros((len(self.states_list[i]), len(self.states_list[i])), dtype=float)
            self.mc_index_list[i] = self.states_list[i].keys()
            self.mc_index_dict[i] = {item: index for index, item in enumerate(self.mc_index_list[i])}
            for curr_state in self.mc_index_list[i]:
                for next_state_pair in self.states_list[i][curr_state]:
                    curr_index = self.mc_index_dict[i][curr_state]
                    next_index = self.mc_index_dict[i][next_state_pair[1]]
                    self.mc_metrix[i][curr_index, next_index] = next_state_pair[0]
            self.mc[i] = MarkovChain(self.mc_metrix[i], list(str(item) for item in self.states_list[i].keys()))

    def __build_all_predict_to_target(self, target_state):
        result_all = []
        inter_matrix = {key: None for key in range(len(self.plan))}
        first_matrix = {key: None for key in range(len(self.plan))}
        for i in range(len(self.plan)):
            combined_array = None
            target_index = self.mc_index_dict[i][target_state]
            max_steps = 0
            for banner_type in self.plan[:i + 1]:
                if banner_type == 0:
                    max_steps += 180
                elif banner_type == 1:
                    max_steps += 231
            for step in range(max_steps):
                if inter_matrix[i] is None:
                    coo_matrix = sp.coo_matrix(self.mc[i].to_matrix())
                    inter_matrix[i] = coo_matrix
                    first_matrix[i] = coo_matrix
                    combined_array = coo_matrix.toarray()[:, target_index]
                else:
                    inter_matrix[i] = first_matrix[i].dot(inter_matrix[i])
                    combined_array = np.vstack((combined_array, inter_matrix[i].toarray()[:, target_index]))
            result_all.append(combined_array.T)
        self.result = result_all

    def _build_calculator(self):
        start_time = time.time()
        self.__build_adj_list()
        time_adj = time.time()
        print("Build adjacent list " + self.wish_model.get_model_name() + " in " + str(time_adj - start_time) + " second(s)")
        self.__build_model()
        time_model = time.time()
        print("Build model " + self.wish_model.get_model_name() + " in " + str(time_model - time_adj) + " second(s)")
        self.__build_all_predict_to_target(self.wish_model.get_empty_model())
        time_result = time.time()
        print("Build result " + self.wish_model.get_model_name() + " in " + str(time_result - time_model) + " second(s)")

        with open('./models/' + self.__gen_model_file_name(), 'wb') as f:
            pickle.dump(self.mc, f)
        with open('./models/' + self.__gen_dict_file_name(), 'wb') as f:
            pickle.dump(self.mc_index_dict, f)
        with open('./models/' + self.__gen_result_file_name(), 'wb') as f:
            pickle.dump(self.result, f)
        print("Build calculator " + self.wish_model.get_model_name() + " in " + str(time.time() - start_time) + " second(s)")

    def get_mc(self):
        return self.mc

    def get_one_predict_from_state(self, start_state, target_state, steps):
        result = [0] * len(self.plan)
        for i in range(len(self.plan)):
            start_state_copy = copy.deepcopy(start_state)
            start_state_copy = start_state_copy.left_n_plan(i + 1)
            start_index = self.mc_index_dict[i][start_state_copy]
            target_index = self.mc_index_dict[i][target_state]
            transition_mat = self.mc[i].to_matrix()
            transition_mat_power = np.linalg.matrix_power(transition_mat, steps)
            result[i] = transition_mat_power[start_index][target_index]
        return result

    def get_all_predict_from_state(self, start_state, target_state, max_steps):
        result_all = []
        inter_matrix = {key: None for key in range(len(self.plan))}
        first_matrix = {key: None for key in range(len(self.plan))}
        inter_start_state = {key: None for key in range(len(self.plan))}
        for i in range(len(self.plan)):
            start_state_copy = copy.deepcopy(start_state)
            inter_start_state[i] = start_state_copy.left_n_plan(i + 1)
        for step in range(max_steps):
            result = [0] * len(self.plan)
            for i in range(len(self.plan)):
                start_index = self.mc_index_dict[i][inter_start_state[i]]
                target_index = self.mc_index_dict[i][target_state]
                if inter_matrix[i] is None:
                    coo_matrix = sp.coo_matrix(self.mc[i].to_matrix())
                    inter_matrix[i] = coo_matrix
                    first_matrix[i] = coo_matrix
                else:
                    inter_matrix[i] = first_matrix[i].dot(inter_matrix[i])
                result[i] = inter_matrix[i].toarray()[start_index][target_index]
            result_all.append(result)
        return result_all


if __name__ == "__main__":
    cal = WishCalculatorV2(model.GenshinWishModel(plan=[0, 0, 0, 0, 0, 0, 0, 1]), force_calculate=False)
    index = cal.mc_index_dict[0][model.GenshinWishModel(plan=[0])]
    print(index)
    print(cal.result[0][index])
