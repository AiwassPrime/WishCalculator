import copy
import os
import pickle
import time

import numpy as np
from matplotlib import pyplot
from pydtmc import MarkovChain, plot_sequence

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

        self.curr_wish_model = None
        self.curr_plan = None
        self.bfs_set = set()
        self.accumulator_lock = threading.Lock()

        model_file_path = './models/' + self.__gen_model_file_name()
        dict_file_path = './models/' + self.__gen_dict_file_name()
        if force_calculate:
            self._build_calculator()
        elif os.path.exists(model_file_path):
            with open(model_file_path, 'rb') as f:
                self.mc = pickle.load(f)
            with open(dict_file_path, 'rb') as f:
                self.mc_index_dict = pickle.load(f)
        else:
            self._build_calculator()

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

    def __gen_model_file_name(self):
        return self.wish_model.get_model_name() + '.pkl'

    def __gen_dict_file_name(self):
        return self.wish_model.get_model_name() + '_dict.pkl'

    def _build_calculator(self):
        start_time = time.time()
        self.__build_adj_list()
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
        with open('./models/' + self.__gen_model_file_name(), 'wb') as f:
            pickle.dump(self.mc, f)
        with open('./models/' + self.__gen_dict_file_name(), 'wb') as f:
            pickle.dump(self.mc_index_dict, f)
        print("Build model " + self.__gen_model_file_name() + " in " + str(time.time() - start_time) + " second(s)")

    def get_mc(self):
        return self.mc


if __name__ == "__main__":
    cal = WishCalculatorV2(model.GenshinWishModel(plan=[0, 1]), force_calculate=True)
    index = cal.mc_index_dict[0][model.GenshinWishModel()]
    distri = np.zeros(len(cal.mc_index_dict[0]), dtype=float)
    distri[index] = 1.0
    mc = cal.get_mc()[0]
    print(mc)
    ret = mc.expected_transitions(1)
    print(ret)
