"""
测试文件：验证 EndfieldCharaWishModel 是否符合卡池规则

规则：
1. 出6星角色基础概率为0.8%，卡池有一个计数器1，在玩家抽第66抽（已经抽了65抽）的时候概率开始增加，
   每抽递增5%概率，第80抽的时候必定出6星角色，出6星角色后该计数器重置为0。计数器1的计数可以从之前的卡池继承。
2. 出6星角色的时候，有50%概率出当期UP角色，另外50%几率出其他6星角色（非目标）
3. 还有一个计数器2，第120抽的时候，如果当前卡池还没有出6星角色，无视以上规则，必定出当期目标6星角色，
   重置计数器1，计数器2仅生效一次
4. 计数器3，独立计数器，每240抽，额外赠送玩家一个当期6星角色（不算抽取，即第240抽的时候玩家可能出任何角色，
   但是同时系统会送玩家一个目标角色）
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calculator.endfield.wish_model_chara import EndfieldCharaWishModel, EndfieldCharaWishModelState
from calculator.endfield import consts


def test_rule1_probability_calculation():
    """测试规则1：概率计算是否正确"""
    print("=" * 60)
    print("测试规则1：概率计算")
    print("=" * 60)
    
    model = EndfieldCharaWishModel(force=True)
    
    # 验证概率数组
    assert len(model.chara) == 90, "概率数组长度应为90"
    
    # 0-65抽：0.8%
    for i in range(66):
        expected = 80 / 10000  # 0.8%
        actual = model.chara[i]
        assert abs(actual - expected) < 0.0001, f"第{i+1}抽概率应为0.8%，实际为{actual*100}%"
        print(f"[OK] 第{i+1}抽概率: {actual*100:.2f}% (期望: 0.8%)")
    
    # 66-79抽：递增5%
    for i in range(66, 80):
        expected = (80 + (500 * (i - 65))) / 10000
        actual = model.chara[i]
        assert abs(actual - expected) < 0.0001, f"第{i+1}抽概率计算错误"
        print(f"[OK] 第{i+1}抽概率: {actual*100:.2f}% (期望: {expected*100:.2f}%)")
    
    # 80抽及以上：100%
    for i in range(80, 90):
        expected = 1.0
        actual = model.chara[i]
        assert abs(actual - expected) < 0.0001, f"第{i+1}抽概率应为100%，实际为{actual*100}%"
        print(f"[OK] 第{i+1}抽概率: {actual*100:.2f}% (期望: 100%)")
    
    print("规则1测试通过！\n")


def test_rule1_counter_reset():
    """测试规则1：出6星后计数器1重置"""
    print("=" * 60)
    print("测试规则1：出6星后计数器1重置")
    print("=" * 60)
    
    model = EndfieldCharaWishModel(force=True)
    
    # 测试状态：已抽79次（下次必定出6星）
    state = EndfieldCharaWishModelState(((79, 0, 0), [0]))
    next_states = model.get_next_states(state)
    
    # 应该有两种可能：50%目标，50%非目标
    # 两种情况下计数器1都应该重置为0
    target_found = False
    pity_found = False
    
    for chance, next_state, is_terminal, result_type in next_states:
        # 验证计数器1重置
        assert next_state[0][0] == 0, f"出6星后计数器1应重置为0，实际为{next_state[0][0]}"
        
        if result_type == consts.EndfieldCharaPullResultType.GET_TARGET:
            target_found = True
            assert abs(chance - 0.5) < 0.01, f"出目标角色概率应为50%，实际为{chance*100}%"
            print(f"[OK] 出目标角色概率: {chance*100:.2f}% (期望: 50%)")
        elif result_type == consts.EndfieldCharaPullResultType.GET_PITY:
            pity_found = True
            assert abs(chance - 0.5) < 0.01, f"出非目标角色概率应为50%，实际为{chance*100}%"
            print(f"[OK] 出非目标角色概率: {chance*100:.2f}% (期望: 50%)")
    
    assert target_found, "应该有可能出目标角色"
    assert pity_found, "应该有可能出非目标角色"
    print("规则1计数器重置测试通过！\n")


def test_rule2_50_percent_split():
    """测试规则2：出6星时50%目标，50%非目标"""
    print("=" * 60)
    print("测试规则2：出6星时50%目标，50%非目标")
    print("=" * 60)
    
    model = EndfieldCharaWishModel(force=True)
    
    # 测试多个状态，确保概率分配正确
    test_cases = [
        ((65, 0, 0), "第66抽，概率开始增加"),
        ((70, 0, 0), "第71抽，概率递增中"),
        ((79, 0, 0), "第80抽，必定出6星"),
    ]
    
    for state_tuple, description in test_cases:
        # 只测试必定出6星的情况（概率>=1）
        if state_tuple[0] >= 79:
            state = EndfieldCharaWishModelState((state_tuple, [0]))
            next_states = model.get_next_states(state)
            
            target_prob = 0.0
            pity_prob = 0.0
            
            for chance, next_state, is_terminal, result_type in next_states:
                if result_type == consts.EndfieldCharaPullResultType.GET_TARGET:
                    target_prob += chance
                elif result_type == consts.EndfieldCharaPullResultType.GET_PITY:
                    pity_prob += chance
            
            # 验证概率总和为1（必定出6星）
            total_prob = target_prob + pity_prob
            assert abs(total_prob - 1.0) < 0.01, f"{description}: 出6星概率总和应为100%，实际为{total_prob*100}%"
            
            # 验证50%分配
            assert abs(target_prob - 0.5) < 0.01, f"{description}: 目标角色概率应为50%，实际为{target_prob*100}%"
            assert abs(pity_prob - 0.5) < 0.01, f"{description}: 非目标角色概率应为50%，实际为{pity_prob*100}%"
            
            print(f"[OK] {description}: 目标={target_prob*100:.2f}%, 非目标={pity_prob*100:.2f}%")
    
    print("规则2测试通过！\n")


def test_rule3_120_pity():
    """测试规则3：120抽必定出目标，仅生效一次"""
    print("=" * 60)
    print("测试规则3：120抽必定出目标，仅生效一次")
    print("=" * 60)
    
    model = EndfieldCharaWishModel(force=True)
    
    # 测试状态：已抽119次，计数器2未使用（z=0）
    state = EndfieldCharaWishModelState(((0, 119, 0), [0]))
    next_states = model.get_next_states(state)
    
    # 应该必定出目标角色
    target_found = False
    total_prob = 0.0
    
    for chance, next_state, is_terminal, result_type in next_states:
        total_prob += chance
        if result_type == consts.EndfieldCharaPullResultType.GET_TARGET:
            target_found = True
            assert abs(chance - 1.0) < 0.01, f"120抽时出目标角色概率应为100%，实际为{chance*100}%"
            # 验证计数器1重置
            assert next_state[0][0] == 0, "120抽保底后计数器1应重置为0"
            # 验证计数器2标记为已使用
            assert next_state[0][2] == 1, "120抽保底后计数器2应标记为已使用"
            print(f"[OK] 120抽保底: 出目标角色概率={chance*100:.2f}%, 计数器1重置={next_state[0][0]}, 计数器2标记={next_state[0][2]}")
    
    assert abs(total_prob - 1.0) < 0.01, "概率总和应为100%"
    assert target_found, "应该必定出目标角色"
    
    # 测试120抽保底已使用后，不再触发
    state_used = EndfieldCharaWishModelState(((0, 119, 1), [0]))
    next_states_used = model.get_next_states(state_used)
    
    # 此时不应再触发120抽保底，应该按正常概率计算
    target_prob_used = 0.0
    for chance, next_state, is_terminal, result_type in next_states_used:
        if result_type == consts.EndfieldCharaPullResultType.GET_TARGET:
            target_prob_used += chance
    
    # 如果计数器1为0，基础概率是0.8%，所以目标概率应该是0.8% * 0.5 = 0.4%
    assert target_prob_used < 0.5, f"120抽保底已使用后，不应再触发保底，目标概率应为正常概率，实际为{target_prob_used*100}%"
    print(f"[OK] 120抽保底已使用后: 目标概率={target_prob_used*100:.2f}% (正常概率)")
    
    print("规则3测试通过！\n")


def test_rule4_240_bonus():
    """测试规则4：每240抽额外赠送目标角色"""
    print("=" * 60)
    print("测试规则4：每240抽额外赠送目标角色")
    print("=" * 60)
    
    model = EndfieldCharaWishModel(force=True)
    
    # 测试状态：计数器2达到239，下次抽卡将达到240
    # 需要找到一个状态，使得下次抽卡后计数器2达到240
    # 状态格式：((计数器1, 计数器2, 120保底标记), 计划列表)
    
    # 测试场景1：第240抽时，如果出目标角色，计划应该减少2个（1个抽取+1个赠送）
    # 注意：如果计数器2>=120，120保底标记必须为1（已使用）
    state = EndfieldCharaWishModelState(((0, 239, 1), [0, 0, 0]))  # 计划有3个目标，120保底已使用
    next_states = model.get_next_states(state)
    
    bonus_found = False
    for chance, next_state, is_terminal, result_type in next_states:
        # 检查是否达到240抽
        if next_state[0][1] == 240:
            if result_type == consts.EndfieldCharaPullResultType.GET_TARGET:
                # 如果出目标角色，且原计划长度>1，应该减少2个
                original_plan_len = len(state[1])
                new_plan_len = len(next_state[1])
                
                if original_plan_len > 1:
                    assert new_plan_len == original_plan_len - 2, \
                        f"240抽时出目标角色，计划应减少2个（原{original_plan_len}->新{new_plan_len}）"
                    bonus_found = True
                    print(f"[OK] 240抽时出目标角色: 计划从{original_plan_len}减少到{new_plan_len}（减少2个）")
                else:
                    # 如果计划只有1个，只能减少1个
                    assert new_plan_len == original_plan_len - 1, \
                        f"240抽时出目标角色但计划只有1个，应减少1个（原{original_plan_len}->新{new_plan_len}）"
                    print(f"[OK] 240抽时出目标角色: 计划从{original_plan_len}减少到{new_plan_len}（减少1个，因为计划只有1个）")
    
    # 测试场景2：第240抽时，如果出非目标角色，计划应该减少1个（只有赠送）
    state2 = EndfieldCharaWishModelState(((79, 239, 1), [0, 0, 0]))  # 计数器1=79，下次必定出6星，120保底已使用
    next_states2 = model.get_next_states(state2)
    
    for chance, next_state, is_terminal, result_type in next_states2:
        if next_state[0][1] == 240:
            if result_type == consts.EndfieldCharaPullResultType.GET_PITY:
                # 出非目标角色，但240抽时系统赠送目标角色
                original_plan_len = len(state2[1])
                new_plan_len = len(next_state[1])
                # 应该减少1个（赠送的目标角色）
                assert new_plan_len == original_plan_len - 1, \
                    f"240抽时出非目标角色，计划应减少1个（赠送）（原{original_plan_len}->新{new_plan_len}）"
                print(f"[OK] 240抽时出非目标角色: 计划从{original_plan_len}减少到{new_plan_len}（减少1个，系统赠送）")
    
    print("规则4测试通过！\n")


def test_counter_inheritance():
    """测试计数器1可以从之前的卡池继承"""
    print("=" * 60)
    print("测试：计数器1可以从之前的卡池继承")
    print("=" * 60)
    
    model = EndfieldCharaWishModel(force=True)
    
    # 测试从不同计数器1值开始的状态
    test_cases = [
        ((0, 0, 0), "从0开始"),
        ((30, 0, 0), "从30继承"),
        ((65, 0, 0), "从65继承（下次开始增加概率）"),
        ((75, 0, 0), "从75继承（概率递增中）"),
    ]
    
    for state_tuple, description in test_cases:
        state = EndfieldCharaWishModelState((state_tuple, [0]))
        next_states = model.get_next_states(state)
        
        # 验证状态可以正常处理
        assert len(next_states) > 0, f"{description}: 应该有下一个状态"
        
        # 验证计数器1正确递增（如果没出6星）
        for chance, next_state, is_terminal, result_type in next_states:
            if result_type == consts.EndfieldCharaPullResultType.GET_NOTHING:
                # 没出6星，计数器1应该+1
                expected_counter = state_tuple[0] + 1
                assert next_state[0][0] == expected_counter, \
                    f"{description}: 没出6星时计数器1应从{state_tuple[0]}变为{expected_counter}，实际为{next_state[0][0]}"
        
        print(f"[OK] {description}: 状态处理正常")
    
    print("计数器继承测试通过！\n")


def test_state_validation():
    """测试状态验证逻辑"""
    print("=" * 60)
    print("测试：状态验证")
    print("=" * 60)
    
    # 测试无效状态：计数器2>=120但标记未使用
    try:
        invalid_state = EndfieldCharaWishModelState(((0, 120, 0), [0]))
        assert False, "应该抛出异常：计数器2>=120但标记未使用"
    except Exception as e:
        assert "invalid" in str(e).lower(), f"应该抛出无效状态异常，实际为: {e}"
        print(f"[OK] 正确拒绝无效状态: {e}")
    
    # 测试有效状态
    valid_states = [
        ((0, 0, 0), [0]),
        ((0, 119, 0), [0]),  # 119抽，未使用120保底
        ((0, 120, 1), [0]),  # 120抽，已使用120保底
        ((79, 239, 1), [0, 0]),  # 239抽，120保底已使用
    ]
    
    for state_tuple, plan in valid_states:
        try:
            state = EndfieldCharaWishModelState((state_tuple, plan))
            print(f"[OK] 有效状态: {state_tuple}, 计划长度={len(plan)}")
        except Exception as e:
            assert False, f"应该接受有效状态{state_tuple}，但抛出异常: {e}"
    
    print("状态验证测试通过！\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始测试 EndfieldCharaWishModel 卡池规则")
    print("=" * 60 + "\n")
    
    try:
        test_rule1_probability_calculation()
        test_rule1_counter_reset()
        test_rule2_50_percent_split()
        test_rule3_120_pity()
        test_rule4_240_bonus()
        test_counter_inheritance()
        test_state_validation()
        
        print("=" * 60)
        print("所有测试通过！[PASS]")
        print("=" * 60)
        return True
    except AssertionError as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n[ERROR] 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
