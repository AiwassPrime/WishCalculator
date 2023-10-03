import pickle
import os

model_file_path = '../models/genshin_v2.pkl'


def is_cache_exist():
    if os.path.exists(model_file_path):
        return True
    return False


class GenshinWishModelV2:
    chara = []
    weapon = []
    chara_cache = {}
    weapon_cache = {}

    def __init__(self):
        if len(self.chara) != 0:
            return
        if is_cache_exist():
            self.__load_cache()
        else:
            self.__fill_chance()
            self.__fill_cache()
            self.__dump_cache()

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
        while len(bfs_queue) > 0:
            curr_process = bfs_queue.pop()
            if self.chara[curr_process[0]] >= 1:
                if curr_process[1] >= 1:
                    chara_cache[curr_process] = (1.0, (0, 0), True)  # get want
                else:
                    chara_cache[curr_process] = (0.5, (0, 0), True)  # get want
                    chara_cache[curr_process] = (0.5, (0, 1), False)  # get pity
                    if (0, 1) not in chara_cache:
                        bfs_queue.append((0, 1))
            else:
                if curr_process[1] >= 1:
                    chara_cache[curr_process] = (self.chara[curr_process[0]], (0, 0), True)  # get want
                    chara_cache[curr_process] = (
                        1.0 - self.chara[curr_process[0]], (curr_process[0] + 1, 1), False)  # get nothing
                    if (curr_process[0] + 1, 1) not in chara_cache:
                        bfs_queue.append((curr_process[0] + 1, 1))
                else:
                    chara_cache[curr_process] = (self.chara[curr_process[0]] * 0.5, (0, 0), True)  # get want
                    chara_cache[curr_process] = (self.chara[curr_process[0]] * 0.5, (0, 1), False)  # get pity
                    chara_cache[curr_process] = (
                        1.0 - self.chara[curr_process[0]], (curr_process[0] + 1, 0), False)  # get nothing
                    if (0, 1) not in chara_cache:
                        bfs_queue.append((0, 1))
                    if (curr_process[0] + 1, 0) not in chara_cache:
                        bfs_queue.append((curr_process[0] + 1, 0))
        weapon_cache = {}
        bfs_queue = [(0, 0, 0)]
        while len(bfs_queue) > 0:
            curr_process = bfs_queue.pop()
            if self.chara[curr_process[0]] >= 1:
                if curr_process[2] >= 2:
                    weapon_cache[curr_process] = (1.0, (0, 0, 0), True)  # get want
                elif curr_process[1] >= 1:
                    weapon_cache[curr_process] = (0.5, (0, 0, 0), True)  # get want
                    weapon_cache[curr_process] = (0.5, (0, 0, curr_process[2] + 1), False)  # get other
                    if (0, 0, curr_process[2] + 1) not in weapon_cache:
                        bfs_queue.append((0, 0, curr_process[2] + 1))
                else:
                    weapon_cache[curr_process] = (0.375, (0, 0, 0), True)  # get want
                    weapon_cache[curr_process] = (0.375, (0, 0, curr_process[2] + 1), False)  # get other
                    weapon_cache[curr_process] = (0.25, (0, 1, curr_process[2] + 1), False)  # get pity
                    if (0, 0, curr_process[2] + 1) not in weapon_cache:
                        bfs_queue.append((0, 0, curr_process[2] + 1))
                    if (0, 1, curr_process[2] + 1) not in weapon_cache:
                        bfs_queue.append((0, 1, curr_process[2] + 1))
            else:
                if curr_process[2] >= 2:
                    weapon_cache[curr_process] = (self.weapon[curr_process[0]], (0, 0, 0), True)  # get want
                    weapon_cache[curr_process] = (
                        1 - self.weapon[curr_process[0]], (curr_process[0] + 1, curr_process[1], 2),
                        False)  # get nothing
                    if (curr_process[0] + 1, curr_process[1], 2) not in weapon_cache:
                        bfs_queue.append((curr_process[0] + 1, curr_process[1], 2))
                elif curr_process[1] >= 1:
                    weapon_cache[curr_process] = (self.weapon[curr_process[0]] * 0.5, (0, 0, 0), True)  # get want
                    weapon_cache[curr_process] = (
                        self.weapon[curr_process[0]] * 0.5, (0, 0, curr_process[2] + 1), False)  # get other
                    weapon_cache[curr_process] = (
                        1.0 - self.weapon[curr_process[0]], (curr_process[0] + 1, 1, curr_process[2]),
                        False)  # get nothing
                    if (0, 0, curr_process[2] + 1) not in weapon_cache:
                        bfs_queue.append((0, 0, curr_process[2] + 1))
                    if (curr_process[0] + 1, 1, curr_process[2]) not in weapon_cache:
                        bfs_queue.append((curr_process[0] + 1, 1, curr_process[2]))
                else:
                    weapon_cache[curr_process] = (self.weapon[curr_process[0]] * 0.375, (0, 0, 0), True)  # get want
                    weapon_cache[curr_process] = (
                        self.weapon[curr_process[0]] * 0.375, (0, 0, curr_process[2] + 1), False)  # get other
                    weapon_cache[curr_process] = (
                        self.weapon[curr_process[0]] * 0.25, (0, 1, curr_process[2] + 1), False)  # get pity
                    weapon_cache[curr_process] = (
                        1.0 - self.weapon[curr_process[0]], (curr_process[0] + 1, 0, curr_process[2]),
                        False)  # get nothing
                    if (0, 0, curr_process[2] + 1) not in weapon_cache:
                        bfs_queue.append((0, 0, curr_process[2] + 1))
                    if (0, 1, curr_process[2] + 1) not in weapon_cache:
                        bfs_queue.append((0, 1, curr_process[2] + 1))
                    if (curr_process[0] + 1, 0, curr_process[2]) not in weapon_cache:
                        bfs_queue.append((curr_process[0] + 1, 0, curr_process[2]))

    def get_next_states(self):
        pass


if __name__ == "__main__":
    pass
