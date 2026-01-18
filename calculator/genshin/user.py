"""
Genshin Impact Wish Calculator - User Module

This module provides the GenshinUser class for managing user state, resources,
and wish calculations in Genshin Impact.
"""

import sys
import os
import copy
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

# Third-party imports
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from loguru import logger

# Local imports
from calculator.genshin import wish_model_v2 as model
from calculator.genshin import consts
from calculator.wish_calculator_v3 import WishCalculatorV3, create_genshin_v2_config

# Add project root to Python path for direct execution
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ============================================================================
# Constants
# ============================================================================

# Resource conversion rates
PRIMOGEM_TO_FATE_RATE = 160  # 160 primogems = 1 intertwined fate
STARGLITTER_TO_FATE_RATE = 5  # 5 starglitter = 1 intertwined fate
CRYSTAL_TO_GEM_RATE = 1  # 1 genesis crystal = 1 primogem


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class GenshinResource:
    """Represents Genshin Impact user resources."""
    intertwined_fate: int
    genesis_crystal: int
    primogem: int
    starglitter: int


# ============================================================================
# Main User Class
# ============================================================================

class GenshinUser:
    """
    Represents a Genshin Impact user with state, resources, and wish calculation capabilities.
    
    Attributes:
        uid: User identifier
        state: Current wish model state
        resource: User's resources (fates, primogems, etc.)
        model: Wish model instance
        calculator: Dictionary of calculators for different states
        prev: Previous user state (for state history)
    """
    
    def __init__(
        self,
        uid: int,
        state: Optional[model.GenshinWishModelState] = None,
        resource: Optional[GenshinResource] = None,
        prev: Optional['GenshinUser'] = None
    ):
        """
        Initialize a Genshin user.
        
        Args:
            uid: User identifier
            state: Initial wish state. If None, creates default state.
            resource: Initial resources. If None, creates empty resources.
            prev: Previous user state for history tracking
        """
        self.uid = uid
        self.state = state or model.GenshinWishModelState(((0, 0), (0, 0, 0), []))
        self.resource = resource or GenshinResource(0, 0, 0, 0)
        self.model = model.GenshinWishModelV2()
        self.calculator: Dict[model.GenshinWishModelState, WishCalculatorV3] = {}
        self.prev = prev

    # ========================================================================
    # Resource Management Methods
    # ========================================================================

    def get_resource(self) -> GenshinResource:
        """Get current user resources."""
        logger.info(f"Get resources for user {self.uid}: {self.resource}")
        return self.resource

    def set_resource(
        self,
        intertwined_fate: int,
        genesis_crystal: int,
        primogem: int,
        starglitter: int
    ) -> None:
        """
        Set user resources.
        
        Args:
            intertwined_fate: Number of intertwined fates
            genesis_crystal: Number of genesis crystals
            primogem: Number of primogems
            starglitter: Number of starglitter
        """
        self.resource = GenshinResource(
            intertwined_fate, genesis_crystal, primogem, starglitter
        )
        logger.info(f"Set resources for user {self.uid}: {self.resource}")

    def update_resource(
        self,
        action: consts.GenshinResourceAction,
        amount: int
    ) -> bool:
        """
        Update resources based on the specified action.
        
        Args:
            action: The resource action to perform
            amount: Amount for the action
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Try to update resources for user {self.uid}: {self.resource}")
        
        action_handlers = {
            consts.GenshinResourceAction.ADJUST_FATE: self._adjust_fate,
            consts.GenshinResourceAction.ADJUST_CRYSTAL: self._adjust_crystal,
            consts.GenshinResourceAction.ADJUST_GEM: self._adjust_gem,
            consts.GenshinResourceAction.ADJUST_STAR: self._adjust_star,
            consts.GenshinResourceAction.FROM_STAR_TO_FATE: self._convert_star_to_fate,
            consts.GenshinResourceAction.FROM_GEM_TO_FATE: self._convert_gem_to_fate,
            consts.GenshinResourceAction.FROM_CRYSTAL_TO_GEM: self._convert_crystal_to_gem,
        }
        
        handler = action_handlers.get(action)
        if handler is None:
            logger.error(
                f"Update resources for user {self.uid} error: "
                f"invalid GenshinResourceAction {action}"
            )
            return False
        
        success = handler(amount)
        if success:
            logger.info(f"Updated resources for user {self.uid}: {self.resource}")
        else:
            logger.warning(
                f"Cannot update resources for user {self.uid} "
                f"with GenshinResourceAction {action}: {self.resource}"
            )
        return success

    def _adjust_fate(self, amount: int) -> bool:
        """Adjust intertwined fate amount."""
        self.resource.intertwined_fate = amount
        return True

    def _adjust_crystal(self, amount: int) -> bool:
        """Adjust genesis crystal amount."""
        self.resource.genesis_crystal = amount
        return True

    def _adjust_gem(self, amount: int) -> bool:
        """Adjust primogem amount."""
        self.resource.primogem = amount
        return True

    def _adjust_star(self, amount: int) -> bool:
        """Adjust starglitter amount."""
        self.resource.starglitter = amount
        return True

    def _convert_star_to_fate(self, amount: int) -> bool:
        """Convert starglitter to intertwined fate."""
        required_starglitter = amount * STARGLITTER_TO_FATE_RATE
        if self.resource.starglitter < required_starglitter:
            return False
        self.resource.intertwined_fate += amount
        self.resource.starglitter -= required_starglitter
        return True

    def _convert_gem_to_fate(self, amount: int) -> bool:
        """Convert primogems to intertwined fate."""
        required_primogem = amount * PRIMOGEM_TO_FATE_RATE
        if self.resource.primogem < required_primogem:
            return False
        self.resource.intertwined_fate += amount
        self.resource.primogem -= required_primogem
        return True

    def _convert_crystal_to_gem(self, amount: int) -> bool:
        """Convert genesis crystal to primogems."""
        if self.resource.genesis_crystal < amount:
            return False
        self.resource.primogem += amount
        self.resource.genesis_crystal -= amount
        return True

    # ========================================================================
    # State Management Methods
    # ========================================================================

    def get_state(self) -> model.GenshinWishModelState:
        """Get current wish state."""
        logger.info(f"Get state for user {self.uid}: {self.state}")
        return self.state

    def set_state(
        self,
        chara_pulls: int,
        chara_pity: consts.GenshinCharaPityType,
        weapon_pulls: int,
        weapon_pity: consts.GenshinWeaponPityType,
        plan: List[consts.GenshinBannerType]
    ) -> bool:
        """
        Set user wish state.
        
        Args:
            chara_pulls: Character banner pull count
            chara_pity: Character pity type
            weapon_pulls: Weapon banner pull count
            weapon_pity: Weapon pity type
            plan: List of banner types in the wish plan
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Try to set state for user {self.uid}: {self.state}")
        
        # Validate character pity type
        valid_chara_pity = (
            consts.GenshinCharaPityType.CHARA_50,
            consts.GenshinCharaPityType.CHARA_100
        )
        if chara_pity not in valid_chara_pity:
            logger.error(
                f"Set state for user {self.uid} error: "
                f"invalid GenshinCharaPityType {chara_pity}"
            )
            return False
        
        # Build character state
        chara_state = (chara_pulls, chara_pity.value)
        
        # Build weapon state
        weapon_state_map = {
            consts.GenshinWeaponPityType.WEAPON_50_PATH_0: (weapon_pulls, 0, 0),
            consts.GenshinWeaponPityType.WEAPON_100_PATH_0: (weapon_pulls, 1, 0),
            consts.GenshinWeaponPityType.WEAPON_50_PATH_1: (weapon_pulls, 0, 1),
            consts.GenshinWeaponPityType.WEAPON_100_PATH_1: (weapon_pulls, 1, 1),
        }
        weapon_state = weapon_state_map.get(weapon_pity)
        if weapon_state is None:
            logger.error(
                f"Set state for user {self.uid} error: "
                f"invalid GenshinWeaponPityType {weapon_pity}"
            )
            return False
        
        # Build plan
        state_plan = []
        valid_banner_types = (
            consts.GenshinBannerType.CHARA,
            consts.GenshinBannerType.WEAPON
        )
        for banner in plan:
            if banner not in valid_banner_types:
                logger.error(
                    f"Set state for user {self.uid} error: "
                    f"invalid GenshinBannerType {banner}"
                )
                return False
            state_plan.append(banner.value)
        
        # Create and set state
        self.state = model.GenshinWishModelState((chara_state, weapon_state, state_plan))
        logger.info(f"Set state for user {self.uid}: {self.state}")
        return True

    def update_state_one_pull(
        self,
        banner: consts.GenshinBannerType,
        pull: consts.GenshinPullResultType
    ) -> Tuple['GenshinUser', bool]:
        """
        Update state after one pull.
        
        Args:
            banner: Banner type for the pull
            pull: Pull result type
            
        Returns:
            Tuple of (new_user, success)
        """
        # Calculate resource cost and update
        updated_resource = self._calculate_pull_cost()
        if updated_resource is None:
            return self, False
        
        # Update state based on banner
        banner_value = banner.value
        
        # Case 1: Pulling on current target banner
        if banner_value == self.state[2][0]:
            return self._update_state_for_current_banner(
                updated_resource, pull
            )
        
        # Case 2: Pulling on a future banner in plan
        elif banner_value in self.state[2]:
            return self._update_state_for_future_banner(
                updated_resource, banner_value, pull
            )
        
        # Case 3: Pulling on a banner not in plan
        else:
            new_user = GenshinUser(
                self.uid,
                state=self.state,
                resource=updated_resource,
                prev=self
            )
            return new_user, True

    def _calculate_pull_cost(self) -> Optional[GenshinResource]:
        """
        Calculate resource cost for one pull.
        
        Returns:
            Updated resource if successful, None otherwise
        """
        updated_resource = copy.deepcopy(self.resource)
        
        # Priority: intertwined_fate > primogem (with crystal supplement) > genesis_crystal
        if updated_resource.intertwined_fate > 0:
            updated_resource.intertwined_fate -= 1
        elif 0 < updated_resource.primogem < PRIMOGEM_TO_FATE_RATE:
            # Supplement primogem with crystal
            needed_crystal = PRIMOGEM_TO_FATE_RATE - updated_resource.primogem
            if updated_resource.genesis_crystal >= needed_crystal:
                updated_resource.genesis_crystal -= needed_crystal
                updated_resource.primogem = 0
            else:
                return None
        elif updated_resource.primogem >= PRIMOGEM_TO_FATE_RATE:
            updated_resource.primogem -= PRIMOGEM_TO_FATE_RATE
        elif updated_resource.genesis_crystal >= PRIMOGEM_TO_FATE_RATE:
            updated_resource.genesis_crystal -= PRIMOGEM_TO_FATE_RATE
        else:
            return None
        
        return updated_resource

    def _update_state_for_current_banner(
        self,
        updated_resource: GenshinResource,
        pull: consts.GenshinPullResultType
    ) -> Tuple['GenshinUser', bool]:
        """Update state when pulling on current target banner."""
        next_states = self.model.get_next_states(self.state)
        
        for next_state in next_states:
            if pull == next_state[3]:
                new_user = GenshinUser(
                    self.uid,
                    state=next_state[1],
                    resource=updated_resource,
                    prev=self
                )
                return new_user, True
        
        return self, False

    def _update_state_for_future_banner(
        self,
        updated_resource: GenshinResource,
        banner_value: int,
        pull: consts.GenshinPullResultType
    ) -> Tuple['GenshinUser', bool]:
        """Update state when pulling on a future banner in plan."""
        # Find banner index and split plan
        banner_index = self.state[2].index(banner_value)
        old_plan = self.state[2][:banner_index]
        new_plan = self.state[2][banner_index:]
        
        # Create temporary state with new plan
        temp_state = model.GenshinWishModelState(
            (self.state[0], self.state[1], new_plan)
        )
        
        # Get next states and find matching pull result
        next_states = self.model.get_next_states(temp_state)
        for next_state in next_states:
            if pull == next_state[3]:
                update_state = next_state[1]
                # Combine old plan with updated state's plan
                final_state = model.GenshinWishModelState((
                    update_state[0],
                    update_state[1],
                    old_plan + update_state[2]
                ))
                new_user = GenshinUser(
                    self.uid,
                    state=final_state,
                    resource=updated_resource,
                    prev=self
                )
                return new_user, True
        
        return self, False

    def update_state_n_pull(
        self,
        banner: consts.GenshinBannerType,
        pull_list: List[consts.GenshinPullResultType]
    ) -> Tuple['GenshinUser', bool]:
        """
        Update state after n pulls.
        
        Args:
            banner: Banner type for the pulls
            pull_list: List of pull result types
            
        Returns:
            Tuple of (new_user, success)
        """
        current_user = copy.deepcopy(self)
        for result in pull_list:
            current_user, success = current_user.update_state_one_pull(banner, result)
            if not success:
                return self, False
        return current_user, True

    # ========================================================================
    # Calculator Methods
    # ========================================================================

    def trigger_calculator(self, force_cpu: bool = False) -> None:
        """
        Trigger calculator for all reduced states.
        
        Args:
            force_cpu: Force CPU usage instead of GPU
        """
        state_list = self.state.get_reduced_state()
        model_config = create_genshin_v2_config()
        for state in state_list:
            self.calculator[state] = WishCalculatorV3(
                self.model, state, model_config, force_cpu=force_cpu
            )

    def get_total_pull(self) -> Tuple[int, float]:
        """
        Calculate total available pulls.
        
        Returns:
            Tuple of (strict_pulls, estimated_pulls)
            - strict_pulls: Exact pulls available
            - estimated_pulls: Estimated pulls (including starglitter conversion)
        """
        strict_pulls = (
            self.resource.intertwined_fate +
            self.resource.primogem // PRIMOGEM_TO_FATE_RATE +
            self.resource.genesis_crystal // PRIMOGEM_TO_FATE_RATE +
            self.resource.starglitter // STARGLITTER_TO_FATE_RATE
        )
        estimated_pulls = strict_pulls / 25 * 26  # Approximation
        return strict_pulls, estimated_pulls

    def get_raw_result(
        self
    ) -> Tuple[Dict[model.GenshinWishModelState, np.ndarray], bool]:
        """
        Get raw calculation results.
        
        Returns:
            Tuple of (result_dict, no_regenerate)
            - result_dict: Dictionary mapping states to result arrays
            - no_regenerate: True if no regeneration needed
        """
        goal_states = self.state.get_reduced_state()[::-1]
        result_list = {}
        no_regenerate = True
        
        calculator_keys = list(self.calculator.keys())[::-1]
        for index, key in enumerate(calculator_keys):
            if index >= len(goal_states):
                break
            
            result, has_result = self.calculator[key].get_result(goal_states[index])
            if has_result:
                no_regenerate = False
                result_list[goal_states[index]] = result
        
        return result_list, no_regenerate


# ============================================================================
# Result Processing Functions
# ============================================================================

def process_result(
    result: Dict[model.GenshinWishModelState, np.ndarray]
) -> Tuple[Dict, List[List[float]], Tuple[int, int]]:
    """
    Process raw result into grid format for visualization.
    
    Args:
        result: Dictionary mapping states to result arrays
        
    Returns:
        Tuple of (result, grid, dimensions)
    """
    num_states = len(result)
    max_length = max(len(res) for res in result.values()) if result else 0
    
    # Initialize grid with 1.0 (default value)
    grid = [[1.0 for _ in range(max_length + 1)] for _ in range(num_states)]
    
    # Fill grid with result values
    for index_x, stats in enumerate(result.values()):
        grid[index_x][0] = 0.0  # First column is 0.0
        for index_y, stat in enumerate(stats):
            grid[index_x][index_y + 1] = stat
    
    return result, grid, (num_states, max_length)


def process_result_agg(
    result: Dict[model.GenshinWishModelState, np.ndarray],
    agg_list: List[float]
) -> Tuple[Dict, List[List[int]], Tuple[int, int]]:
    """
    Process raw result with aggregation thresholds.
    
    Args:
        result: Dictionary mapping states to result arrays
        agg_list: List of aggregation thresholds (e.g., [0.1, 0.25, 0.5, 0.75, 0.9])
        
    Returns:
        Tuple of (agg_result, grid, dimensions)
    """
    agg_result = {}
    num_states = len(result)
    max_length = max(len(res) for res in result.values()) if result else 0
    
    # Calculate aggregation points for each state
    for state, res in result.items():
        agg_res = [
            np.searchsorted(res, agg_num, side='right')
            for agg_num in agg_list
        ]
        agg_result[state] = agg_res
    
    # Create grid marking aggregation points
    grid = [[0 for _ in range(max_length + 1)] for _ in range(num_states)]
    for index, agg_stats in enumerate(agg_result.values()):
        for stat in agg_stats:
            grid[index][stat + 1] = 1
    
    return agg_result, grid, (num_states, max_length)


# ============================================================================
# Utility Functions
# ============================================================================

def init_genshin_user(user_name: str, passcode: str) -> GenshinUser:
    """
    Initialize a Genshin user (placeholder for future authentication).
    
    Args:
        user_name: User name
        passcode: Passcode (not currently used)
        
    Returns:
        New GenshinUser instance
    """
    return GenshinUser(hash(user_name))


def show_graph():
    matplotlib.use('qtagg')

    # Initialize user
    user = GenshinUser(1)

    # Set resources
    user.update_resource(consts.GenshinResourceAction.ADJUST_FATE, 841)
    user.update_resource(consts.GenshinResourceAction.ADJUST_GEM, 0)
    user.update_resource(consts.GenshinResourceAction.ADJUST_CRYSTAL, 0)
    user.update_resource(consts.GenshinResourceAction.ADJUST_STAR, 0)
    pull, pull_est = user.get_total_pull()

    # Set state
    user.set_state(
        0,
        consts.GenshinCharaPityType.CHARA_50,
        0,
        consts.GenshinWeaponPityType.WEAPON_50_PATH_0,
        [
            consts.GenshinBannerType.CHARA,
            consts.GenshinBannerType.CHARA,
            consts.GenshinBannerType.CHARA,
            consts.GenshinBannerType.CHARA,
            consts.GenshinBannerType.CHARA,
            consts.GenshinBannerType.CHARA,
            consts.GenshinBannerType.CHARA,
            consts.GenshinBannerType.WEAPON
        ]
    )

    # Trigger calculator
    user.trigger_calculator(force_cpu=False)

    # Get and process results
    raw, is_success = user.get_raw_result()
    _, graph, dem = process_result(raw)
    agg, _, _ = process_result_agg(raw, [0.1, 0.25, 0.5, 0.75, 0.9])

    # Visualization setup
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=1)

    plt.figure(figsize=(16, 9))
    plt.imshow(graph, cmap=cmap, interpolation='none', aspect='auto', norm=norm)

    # Draw pull indicators
    plt.fill_betweenx(
        y=[-0.5, len(agg) - 0.5],
        x1=pull,
        x2=pull,
        color='blue'
    )
    plt.fill_betweenx(
        y=[-0.5, len(agg) - 0.5],
        x1=pull_est,
        x2=pull_est,
        color='blue',
        alpha=0.2
    )

    # Draw aggregation markers
    for index, agg_stats in enumerate(agg.values()):
        for stats in agg_stats:
            plt.fill_betweenx(
                y=[index - 0.50, index + 0.49],
                x1=stats,
                x2=stats,
                color='black'
            )

    # Colorbar
    cbar = plt.colorbar()
    cbar.set_label('Values')

    # Axis labels
    x_indices = np.arange(0, dem[1], 100)
    x_labels = [str(index) for index in x_indices]
    y_indices = np.arange(0, dem[0], 1)
    y_labels = [agg_s.get_plan_str() for agg_s in agg.keys()]
    plt.xticks(ticks=x_indices, labels=x_labels)
    plt.yticks(ticks=y_indices, labels=y_labels)

    # Pull count label
    plt.text(
        pull, -0.8, f'You={pull}',
        color='blue', fontsize=12, ha='center', va='center'
    )

    # Interactive vertical line
    ax = plt.gca()
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    initial_x = (x_min + x_max) / 2
    vertical_line = ax.axvline(
        x=initial_x, color='red', linewidth=2,
        linestyle='--', alpha=0.7
    )
    text_label = ax.text(
        initial_x, y_max * 0.95, f'x = {int(initial_x)}',
        color='red', fontsize=12, ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # Drag state
    drag_state = {'is_dragging': False}

    def on_press(event):
        """Handle mouse press event."""
        if event.inaxes != ax:
            return
        line_x = vertical_line.get_xdata()[0]
        tolerance = (x_max - x_min) * 0.02  # 2% tolerance
        if abs(event.xdata - line_x) < tolerance:
            drag_state['is_dragging'] = True

    def on_motion(event):
        """Handle mouse motion event."""
        if not drag_state['is_dragging'] or event.inaxes != ax:
            return
        if event.xdata is None:
            return
        x_pos = max(x_min, min(x_max, event.xdata))
        vertical_line.set_xdata([x_pos, x_pos])
        text_label.set_x(x_pos)
        text_label.set_text(f'x = {int(x_pos)}')
        plt.draw()

    def on_release(event):
        """Handle mouse release event."""
        drag_state['is_dragging'] = False

    # Connect events
    fig = plt.gcf()
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)

    plt.show()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    show_graph()
    