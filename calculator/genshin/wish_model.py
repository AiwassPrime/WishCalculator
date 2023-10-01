class GenshinWishModel:
    chara = None
    weapon = None
    chara_cache = {}
    weapon_cache = {}

    def __init__(self, state=((0, 0), (0, 0, 0)), plan=None):
        if plan is None:
            plan = []
        if self.chara is None:
            chara_list = []
            for i in range(100):
                if i <= 73:
                    chara_list.append(60 / 10000)
                else:
                    chara_list.append((60 + (600 * (i - 73))) / 10000)
            self.chara = chara_list
        if self.weapon is None:
            weapon_list = []
            for i in range(100):
                if i <= 62:
                    weapon_list.append(70 / 10000)
                else:
                    weapon_list.append((70 + (700 * (i - 62))) / 10000)
            self.weapon = weapon_list

        self.state = state
        self.plan = plan

    def __hash__(self):
        return hash((self.state, len(self.plan)))

    def __eq__(self, other):
        if not hasattr(other, 'state'):
            return False
        if not hasattr(other, 'plan'):
            return False
        return self.state == other.state and len(self.plan) == len(other.plan)

    def __str__(self):
        details = str((self.state, len(self.plan)))
        return details

    def left_n_plan(self, n):
        return GenshinWishModel(plan=self.plan[:n])

    def del_last_plan(self):
        return GenshinWishModel(plan=self.plan[:-1])

    def get_state(self):
        return self.state

    def get_plan(self):
        return self.plan

    def get_model_name(self):
        binary_string = ''.join(str(num) for num in self.plan)
        return "genshin_" + str(hash(self.state)) + "_" + binary_string

    def is_win(self, fulfillment=0):
        if len(self.plan) <= fulfillment:
            return True
        else:
            return False

    def get_next_states(self, iter_chance=1.0):
        if len(self.plan) <= 0:
            return [(iter_chance, self)]
        if self.plan[0] == 0:
            if self.state in self.chara_cache:
                pairs = self.chara_cache[self.state]
                result = list(map(lambda pair: (pair[0] * iter_chance,
                                                GenshinWishModel(state=pair[1], plan=self.plan[1:])
                                                if pair[2]
                                                else GenshinWishModel(state=pair[1], plan=self.plan)),
                                  pairs))
                return result
            get_chance = self.chara[self.state[0][0] + 1]
            if get_chance > 1:
                get_chance = 1
            no_get_chance = 1 - get_chance
            if self.state[0][1] == 0:
                get_chance_half = 0.5 * get_chance
                get_chara = GenshinWishModel(state=((0, 0), self.state[1]),
                                             plan=self.plan[1:])
                get_pity = GenshinWishModel(state=((0, 1), self.state[1]),
                                            plan=self.plan)
                get_nothing = GenshinWishModel(state=((self.state[0][0] + 1, 0), self.state[1]), plan=self.plan)
                if get_chance >= 1:
                    self.chara_cache[self.state] = [(get_chance_half, get_chara.get_state(), True),
                                                    (get_chance_half, get_pity.get_state(), False)]
                    return [(get_chance_half*iter_chance, get_chara), (get_chance_half*iter_chance, get_pity)]
                else:
                    self.chara_cache[self.state] = [(get_chance_half, get_chara.get_state(), True),
                                                    (get_chance_half, get_pity.get_state(), False),
                                                    (no_get_chance, get_nothing.get_state(), False)]
                    return [(get_chance_half*iter_chance, get_chara), (get_chance_half*iter_chance, get_pity),
                            (no_get_chance*iter_chance, get_nothing)]
            elif self.state[0][1] == 1:
                get_chara = GenshinWishModel(state=((0, 0), self.state[1]),
                                             plan=self.plan[1:])
                get_nothing = GenshinWishModel(state=((self.state[0][0] + 1, 1), self.state[1]), plan=self.plan)
                if get_chance >= 1:
                    self.chara_cache[self.state] = [(get_chance, get_chara.get_state(), True)]
                    return [(get_chance*iter_chance, get_chara)]
                else:
                    self.chara_cache[self.state] = [(get_chance, get_chara.get_state(), True),
                                                    (no_get_chance, get_nothing.get_state(), False)]
                    return [(get_chance*iter_chance, get_chara), (no_get_chance*iter_chance, get_nothing)]
            else:
                raise Exception("Unexpected state: " + str(self.state[0]))
        elif self.plan[0] == 1:
            if self.state in self.weapon_cache:
                pairs = self.weapon_cache[self.state]
                result = list(map(lambda pair: (pair[0] * iter_chance,
                                                GenshinWishModel(state=pair[1], plan=self.plan[1:])
                                                if pair[2]
                                                else GenshinWishModel(state=pair[1], plan=self.plan)),
                                  pairs))
                return result
            get_chance = self.weapon[self.state[1][0] + 1]
            if get_chance > 1:
                get_chance = 1
            no_get_chance = 1 - get_chance
            if self.state[1][1] == 0 and self.state[1][2] < 2:
                get_up_chance = 0.375 * get_chance
                get_pity_chance = 0.25 * get_chance
                get_weapon = GenshinWishModel(state=(self.state[0], (0, 0, 0)), plan=self.plan[1:])
                get_other_weapon = GenshinWishModel(state=(self.state[0], (0, 0, self.state[1][2]+1)), plan=self.plan)
                get_pity = GenshinWishModel(state=(self.state[0], (0, 1, self.state[1][2]+1)), plan=self.plan)
                get_nothing = GenshinWishModel(state=(self.state[0], (self.state[1][0]+1, 0, self.state[1][2])),
                                               plan=self.plan)
                if get_chance >= 1:
                    self.weapon_cache[self.state] = [(get_up_chance, get_weapon.get_state(), True),
                                                     (get_up_chance, get_other_weapon.get_state(), False),
                                                     (get_pity_chance, get_pity.get_state(), False)]
                    return [(get_up_chance*iter_chance, get_weapon),
                            (get_up_chance*iter_chance, get_other_weapon),
                            (get_pity_chance*iter_chance, get_pity)]
                else:
                    self.weapon_cache[self.state] = [(get_up_chance, get_weapon.get_state(), True),
                                                     (get_up_chance, get_other_weapon.get_state(), False),
                                                     (get_pity_chance, get_pity.get_state(), False),
                                                     (no_get_chance, get_nothing.get_state(), False)]
                    return [(get_up_chance*iter_chance, get_weapon),
                            (get_up_chance*iter_chance, get_other_weapon),
                            (get_pity_chance*iter_chance, get_pity),
                            (no_get_chance*iter_chance, get_nothing)]
            elif self.state[1][1] == 1 and self.state[1][2] < 2:
                get_up_chance = 0.5 * get_chance
                get_weapon = GenshinWishModel(state=(self.state[0], (0, 0, 0)), plan=self.plan[1:])
                get_other_weapon = GenshinWishModel(state=(self.state[0], (0, 0, self.state[1][2]+1)), plan=self.plan)
                get_nothing = GenshinWishModel(state=(self.state[0], (self.state[1][0] + 1, 1, self.state[1][2])),
                                               plan=self.plan)
                if get_chance >= 1:
                    self.weapon_cache[self.state] = [(get_up_chance, get_weapon.get_state(), True),
                                                     (get_up_chance, get_other_weapon.get_state(), False)]
                    return [(get_up_chance*iter_chance, get_weapon), (get_up_chance*iter_chance, get_other_weapon)]
                else:
                    self.weapon_cache[self.state] = [(get_up_chance, get_weapon.get_state(), True),
                                                     (get_up_chance, get_other_weapon.get_state(), False),
                                                     (no_get_chance, get_nothing.get_state(), False)]
                    return [(get_up_chance*iter_chance, get_weapon),
                            (get_up_chance * iter_chance, get_other_weapon),
                            (no_get_chance*iter_chance, get_nothing)]
            elif self.state[1][2] >= 2:
                get_weapon = GenshinWishModel(state=(self.state[0], (0, 0, 0)), plan=self.plan[1:])
                get_nothing = GenshinWishModel(state=(self.state[0],
                                                      (self.state[1][0] + 1, self.state[1][1], self.state[1][2])),
                                               plan=self.plan)
                if get_chance >= 1:
                    self.weapon_cache[self.state] = [(get_chance, get_weapon.get_state(), True)]
                    return [(get_chance*iter_chance, get_weapon)]
                else:
                    self.weapon_cache[self.state] = [(get_chance, get_weapon.get_state(), True),
                                                     (no_get_chance, get_nothing.get_state(), False)]
                    return [(get_chance*iter_chance, get_weapon), (no_get_chance*iter_chance, get_nothing)]
            else:
                raise Exception("Unexpected state: " + str(self.state[1]))
        else:
            raise Exception("Unexpected plan: " + str(self.plan))


if __name__ == "__main__":
    model = GenshinWishModel(state=((0, 0), (0, 0, 0)), plan=[1])
    l0 = model.get_next_states()
    l1 = l0[0][1].get_next_states()
    l2 = l0[1][1].get_next_states()
    l3 = l0[2][1].get_next_states()
