import copy
from loguru import logger
import os
import pickle
import time
import numpy as np
import scipy.sparse as sp
from calculator.definitions import ROOT_DIR
from typing import Callable, Optional


class ModelConfig:
    """模型配置类，包含所有模型特定的配置"""
    def __init__(
        self,
        model_name: str,
        max_steps_calculator: Callable,
        cache_file_name_generator: Optional[Callable] = None
    ):
        """
        初始化模型配置
        
        Args:
            model_name: 模型名称，用于生成缓存文件名
            max_steps_calculator: 计算最大步数的函数，接收 init_state 作为参数，返回 int
            cache_file_name_generator: 可选，生成缓存文件名的函数。如果为 None，使用默认格式：
                                      '{model_name}_{state_hash}.pkl'
        """
        self.model_name = model_name
        self.max_steps_calculator = max_steps_calculator
        self.cache_file_name_generator = cache_file_name_generator or (
            lambda state, model_name: f'{model_name}_{state.gen_base64()}.pkl'
        )


class WishCalculatorV3:
    def __init__(
        self,
        wish_model,
        init_state,
        model_config: ModelConfig,
        force=False,
        force_cpu=False
    ):
        """
        初始化 WishCalculatorV3
        
        Args:
            wish_model: 愿望模型对象，需要有 get_next_states(state) 方法
            init_state: 初始状态对象，需要有 get_goal_state() 方法
            model_config: 模型配置对象，包含模型特定的配置
            force: 是否强制重新构建模型
            force_cpu: 是否强制使用 CPU
        """
        self.model = wish_model
        self.init_state = init_state
        self.model_config = model_config
        
        # 生成缓存文件路径
        cache_filename = self.model_config.cache_file_name_generator(
            self.init_state,
            self.model_config.model_name
        )
        self.model_file_path = os.path.join(ROOT_DIR, 'models', cache_filename)

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

        # 使用配置中的 max_steps_calculator 计算最大步数
        max_steps = self.model_config.max_steps_calculator(self.init_state)

        goal_states = self.init_state.get_goal_state()[-1]
        target_index = [self.adjacency_matrix_index[s] for s in goal_states]

        try:
            import cupy as cp
        except ImportError:
            force_cpu = True
        if force_cpu or cp.cuda.runtime.getDeviceCount() == 0:
            logger.info("Use CPU")
            result = np.zeros((len(self.adjacency_matrix_index), max_steps), dtype=float)
            coo_matrix = sp.coo_matrix(self.adjacency_matrix)
            inter_matrix = None
            first_matrix = copy.deepcopy(coo_matrix)
            for step in range(max_steps):
                logger.debug("Step " + str(step))
                if inter_matrix is None:
                    inter_matrix = coo_matrix
                    target_arrays = inter_matrix.toarray()[:, target_index].sum(axis=1)
                    result[:, step] = target_arrays
                else:
                    inter_matrix = first_matrix.dot(inter_matrix)
                    target_arrays = inter_matrix.toarray()[:, target_index].sum(axis=1)
                    result[:, step] = target_arrays
        else:
            logger.info("Use GPU")
            result = cp.zeros((len(self.adjacency_matrix_index), max_steps), dtype=float)
            inter_matrix = cp.sparse.csr_matrix(cp.asarray(self.adjacency_matrix))
            first_matrix = copy.deepcopy(inter_matrix)
            for step in range(max_steps):
                logger.debug("Step " + str(step))
                if step == 0:
                    target_arrays = inter_matrix.toarray()[:, target_index].sum(axis=1)
                else:
                    inter_matrix = first_matrix.dot(inter_matrix)
                    target_arrays = inter_matrix.toarray()[:, target_index].sum(axis=1)
                result[:, step] = target_arrays
            result = cp.asnumpy(result)

        self.result = result

        logger.info("Build model in " + str(time.time() - start_time) + " second(s)")

    def get_result(self, start_state):
        if start_state not in self.adjacency_matrix_index:
            return None, False
        else:
            return self.result[self.adjacency_matrix_index[start_state]], True


# 辅助函数：为不同模型创建配置
#
# 使用示例：
#   # 为 Genshin V2 模型创建配置
#   config = create_genshin_v2_config()
#   calculator = WishCalculatorV3(model, init_state, config)
#
#   # 为其他模型创建配置（例如 Endfield）
#   def endfield_max_steps_calculator(init_state):
#       # 根据 Endfield 的规则计算最大步数
#       return len(init_state[1]) * 240  # 示例：每个目标最多240抽
#
#   endfield_config = ModelConfig(
#       model_name='endfield_chara',
#       max_steps_calculator=endfield_max_steps_calculator
#   )
#   calculator = WishCalculatorV3(endfield_model, endfield_state, endfield_config)

def create_genshin_v2_config() -> ModelConfig:
    """
    创建 Genshin V2 模型的配置
    
    Returns:
        ModelConfig: Genshin V2 模型的配置对象
    """
    def max_steps_calculator(init_state):
        """计算 Genshin V2 模型的最大步数"""
        max_steps = 0
        for banner_type in init_state[2]:
            if banner_type == 0:
                max_steps += 180  # 角色池
            elif banner_type == 1:
                max_steps += 231  # 武器池
        return max_steps
    
    def cache_file_name_generator(state, model_name):
        """生成 Genshin V2 模型的缓存文件名"""
        return f'{model_name}_{state.gen_base64()}.pkl'
    
    return ModelConfig(
        model_name='genshin_v2',
        max_steps_calculator=max_steps_calculator,
        cache_file_name_generator=cache_file_name_generator
    )


def create_endfield_chara_config() -> ModelConfig:
    """
    创建 Endfield Chara 模型的配置

    Returns:
        ModelConfig: Endfield Chara 模型的配置对象
    """

    def max_steps_calculator(init_state):
        """计算 Endfield Chara 模型的最大步数"""
        max_steps = 0
        if len(init_state[1]) <= 1:
            return 120
        elif len(init_state[1]) == 2:
            return 240
        else:
            return 240 * len(init_state[1]) - 1

    def cache_file_name_generator(state, model_name):
        """生成 Endfield Chara 模型的缓存文件名"""
        return f'{model_name}_{state.gen_base64()}.pkl'

    return ModelConfig(
        model_name='endfield_chara',
        max_steps_calculator=max_steps_calculator,
        cache_file_name_generator=cache_file_name_generator
    )


if __name__ == "__main__":
    pass
