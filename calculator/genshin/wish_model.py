class GenshinWishModel:
    def __init__(self, chara, weapon, state=((0, 0), (0, 0, 0)), plan=None):
        if plan is None:
            plan = []
        self.state = state
        self.plan = plan
        self.chara = chara
        self.weapon = weapon

    def input_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def input_plan(self, plan):
        self.plan = plan

    def get_plan(self):
        return self.plan

    def is_win(self):
        if len(self.plan) <= 0:
            return True
        else:
            return False

    def get_next_states(self):
        if self.plan[0] == 0:
            get_chance = self.chara[self.state[0][0]]
            if get_chance > 1:
                get_chance = 1
            no_get_chance = 1 - get_chance
            if self.state[0][1] == 0:
                get_chance_half = 0.5 * get_chance
                get_chara = GenshinWishModel(self.chara, self.weapon, state=((0, 0), self.state[1]), plan=self.plan[1:])
                get_pity = GenshinWishModel(self.chara, self.weapon, state=((0, 1), self.state[1]), plan=self.plan)
                get_nothing = GenshinWishModel(self.chara, self.weapon,
                                               state=((self.state[0][0]+1, 0), self.state[1]), plan=self.plan)
                if get_chance >= 1:
                    return [(get_chance_half, get_chara), (get_chance_half, get_pity)]
                else:
                    return [(get_chance_half, get_chara), (get_chance_half, get_pity), (no_get_chance, get_nothing)]
            elif self.state[0][1] == 1:
                get_chara = GenshinWishModel(self.chara, self.weapon, state=((0, 0), self.state[1]), plan=self.plan[1:])
                get_nothing = GenshinWishModel(self.chara, self.weapon,
                                               state=((self.state[0][0]+1, 1), self.state[1]), plan=self.plan)
                if get_chance >= 1:
                    return [(get_chance, get_chara)]
                else:
                    return [(get_chance, get_chara), (no_get_chance, get_nothing)]
            else:
                raise Exception("Unexpected state: " + str(self.state[0]))
        elif self.plan[0] == 0:
            get_chance = self.weapon[self.state[1][0]]
            if get_chance > 1:
                get_chance = 1
            no_get_chance = 1 - get_chance
            if self.state[1][1] == 0 and self.state[1][2] < 2:
                get_chance_reduction = 0.375 * get_chance
                get_weapon = GenshinWishModel(self.chara, self.weapon,
                                              state=(self.state[0], (0, 0, 0)), plan=self.plan[1:])
                get_pity = GenshinWishModel(self.chara, self.weapon,
                                            state=(self.state[0], (0, 1, self.state[1][2] + 1)), plan=self.plan)
                get_nothing = GenshinWishModel(self.chara, self.weapon,
                                               state=(self.state[0], (self.state[1][0] + 1, 0, self.state[1][2])),
                                               plan=self.plan)
                if get_chance >= 1:
                    return [(get_chance_reduction, get_weapon), (get_chance_reduction, get_pity)]
                else:
                    return [(get_chance_reduction, get_weapon), (get_chance_reduction, get_pity),
                            (no_get_chance, get_nothing)]
            elif self.state[1][1] == 1 and self.state[1][2] < 2:
                get_chance_reduction = 0.75 * get_chance
                get_weapon = GenshinWishModel(self.chara, self.weapon,
                                              state=(self.state[0], (0, 0, 0)), plan=self.plan[1:])
                get_nothing = GenshinWishModel(self.chara, self.weapon,
                                               state=(self.state[0],
                                                      (self.state[1][0] + 1, 1, self.state[1][2])),
                                               plan=self.plan)
                if get_chance >= 1:
                    return [(get_chance_reduction, get_weapon)]
                else:
                    return [(get_chance_reduction, get_weapon), (no_get_chance, get_nothing)]
            elif self.state[1][2] >= 2:
                get_chance_reduction = 0.35 * get_chance
                get_weapon = GenshinWishModel(self.chara, self.weapon,
                                              state=(self.state[0], (0, 0, 0)), plan=self.plan[1:])
                get_nothing = GenshinWishModel(self.chara, self.weapon,
                                               state=(self.state[0],
                                                      (self.state[1][0] + 1, self.state[1][1], self.state[1][2])),
                                               plan=self.plan)
                if get_chance >= 1:
                    return [(get_chance_reduction, get_weapon)]
                else:
                    return [(get_chance_reduction, get_weapon), (no_get_chance, get_nothing)]
            else:
                raise Exception("Unexpected state: " + str(self.state[1]))
        else:
            raise Exception("Unexpected plan: " + str(self.plan))


if __name__ == "__main__":
    charaList = []
    for i in range(100):
        if i <= 73:
            charaList.append(60 / 10000)
        else:
            charaList.append((60 + (600 * (i - 73))) / 10000)
    weaponList = []
    for i in range(100):
        if i <= 62:
            weaponList.append(70 / 10000)
        else:
            weaponList.append((70 + (700 * (i - 62))) / 10000)
    model = GenshinWishModel(charaList, weaponList, plan=[0])
    print(model.get_state())
    print(model.get_next_states())
