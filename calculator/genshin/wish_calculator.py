import time

import wish_model
import copy
import concurrent.futures
import threading


class GenshinWishCalculator:
    def __init__(self, state, plan):
        self.model = wish_model.GenshinWishModel(state=state, plan=plan)
        self.plan = plan

        self.iter = 0
        self.curr_iter = {self.model: 1.0}
        self.win_chance = [0] * len(plan)
        self.result = [self.win_chance]

        self.iter_limit = 240 * len(plan)
        self.accumulator_lock = threading.Lock()

    def __process_model(self, model, chance):
        res = model.get_next_states(iter_chance=chance)
        for next_chance, next_model in res:
            with self.accumulator_lock:
                if next_model in self.curr_iter:
                    self.curr_iter[next_model] += next_chance
                else:
                    self.curr_iter[next_model] = next_chance
                for i in range(len(self.plan)):
                    if next_model.is_win(fulfillment=i):
                        self.win_chance[i] += next_chance

    def one_iter(self):
        start_time = time.time()
        curr = copy.deepcopy(self.curr_iter)
        self.win_chance = [0] * len(self.plan)
        self.curr_iter = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Use executor.map() to apply the function to dictionary items in parallel
            executor.map(self.__process_model, curr.keys(), curr.values())
        self.iter += 1
        self.result.append(self.win_chance)
        print("Iter " + str(self.iter) + " used " + str(time.time() - start_time) + " second(s)")

    def calculate(self):
        while self.iter <= self.iter_limit:
            self.one_iter()


if __name__ == "__main__":
    cal = GenshinWishCalculator(((0, 0), (0, 0, 0)), [0, 0, 0, 0, 0, 0, 0])
    cal.calculate()
    print(cal.result)
