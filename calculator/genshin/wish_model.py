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
            get = self.chara[self.state[0][0]]
            no_get = 1 - get
            if self.state[0][0] == 0:
                get_half = 0.5 * get
                get_chara = GenshinWishModel(self.chara, self.weapon, state=((0, 0), self.state[1]), plan=self.plan[1:])
                get_pity = GenshinWishModel(self.chara, self.weapon, state=((0, 1), self.state[1]), plan=self.plan)
                get_nothing = GenshinWishModel(self.chara, self.weapon, state=((self.state[0][0]+1, 0), self.state[1]), plan=self.plan)
                return [(get_half, get_chara), (get_half, get_pity), (no_get, get_nothing)]
            else:
                get_chara = GenshinWishModel(self.chara, self.weapon, state=((0, 0), self.state[1]), plan=self.plan[1:])
                get_nothing = GenshinWishModel(self.chara, self.weapon, state=((self.state[0][0]+1, 1), self.state[1]), plan=self.plan)
                return [(get, get_chara), (no_get, get_nothing)]
        else:
            get = 0

