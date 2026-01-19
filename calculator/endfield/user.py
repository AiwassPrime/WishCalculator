"""
Endfield Wish Calculator - User Module

This module provides the EndfieldUser class for managing user state, resources,
and wish calculations in Endfield.
"""
import platform
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
from calculator.endfield import wish_model_chara as model
from calculator.endfield import consts
from calculator.wish_calculator_v3 import WishCalculatorV3, create_endfield_chara_config

# Add project root to Python path for direct execution
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ============================================================================
# Constants
# ============================================================================

# Resource conversion rates


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class EndfieldCharaResource:
    """Represents Endfield user resources."""
    dummy: int


# ============================================================================
# Main User Class
# ============================================================================

class EndfieldUser:
    """
    Represents an Endfield user with state, resources, and wish calculation capabilities.

    Attributes:
        uid: User identifier
        state: Current wish model state
        resource: User's resources (dummy, etc.)
        model: Wish model instance
        calculator: Dictionary of calculators for different states
        prev: Previous user state (for state history)
    """

    def __init__(
            self,
            uid: int,
            state: Optional[model.EndfieldCharaWishModelState] = None,
            resource: Optional[EndfieldCharaResource] = None,
            prev: Optional['EndfieldUser'] = None
    ):
        """
        Initialize a Endfield user.

        Args:
            uid: User identifier
            state: Initial wish state. If None, creates default state.
            resource: Initial resources. If None, creates empty resources.
            prev: Previous user state for history tracking
        """
        self.uid = uid
        self.state = state or model.EndfieldCharaWishModelState(((0, 0, 0), [0]))
        self.resource = resource or EndfieldCharaResource(0)
        self.model = model.EndfieldCharaWishModel(have_30_extra=False)
        self.calculator: Dict[model.EndfieldCharaWishModelState, WishCalculatorV3] = {}
        self.prev = prev

    # ========================================================================
    # Resource Management Methods
    # ========================================================================

    def get_resource(self) -> EndfieldCharaResource:
        """Get current user resources."""
        logger.info(f"Get resources for user {self.uid}: {self.resource}")
        return self.resource

    def set_resource(
            self,
            dummy: int,
    ) -> None:
        """
        Set user resources.

        Args:
            dummy: Number of dummy
        """
        self.resource = EndfieldCharaResource(
            dummy
        )
        logger.info(f"Set resources for user {self.uid}: {self.resource}")

    def update_resource(
            self,
            action: consts.EndfieldResourceAction,
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
            consts.EndfieldResourceAction.ADJUST_DUMMY: self._adjust_dummy,
        }

        handler = action_handlers.get(action)
        if handler is None:
            logger.error(
                f"Update resources for user {self.uid} error: "
                f"invalid EndfieldResourceAction {action}"
            )
            return False

        success = handler(amount)
        if success:
            logger.info(f"Updated resources for user {self.uid}: {self.resource}")
        else:
            logger.warning(
                f"Cannot update resources for user {self.uid} "
                f"with EndfieldResourceAction {action}: {self.resource}"
            )
        return success

    def _adjust_dummy(self, amount: int) -> bool:
        """Adjust intertwined fate amount."""
        self.resource.dummy = amount
        return True


    # ========================================================================
    # State Management Methods
    # ========================================================================

    def get_state(self) -> model.EndfieldCharaWishModelState:
        """Get current wish state."""
        logger.info(f"Get state for user {self.uid}: {self.state}")
        return self.state

    def set_state(
            self,
            pulls_after_6_star: int,
            pulls_in_banner: int,
            have_120: bool,
            plan: List[consts.EndfieldBannerType]
    ) -> bool:
        logger.info(f"Try to set state for user {self.uid}: {self.state}")

        # Build character state
        chara_state = (pulls_after_6_star, pulls_in_banner, 0 if have_120 else 1)

        # Build plan
        state_plan = []
        for banner in plan:
            state_plan.append(banner.value)

        # Create and set state
        self.state = model.EndfieldCharaWishModelState((chara_state, state_plan))
        logger.info(f"Set state for user {self.uid}: {self.state}")
        return True


    # ========================================================================
    # Calculator Methods
    # ========================================================================

    def trigger_calculator(self, force_cpu: bool = False, use_dense: bool = False) -> None:
        """
        Trigger calculator for all reduced states.

        Args:
            force_cpu: Force CPU usage instead of GPU
            use_dense: Use dense array calculation
        """
        state_list = self.state.get_reduced_state()
        model_config = create_endfield_chara_config()
        for state in state_list:
            self.calculator[state] = WishCalculatorV3(
                self.model, state, model_config, force_cpu=force_cpu, use_dense=use_dense
            )

    def get_total_pull(self) -> Tuple[int, float]:
        strict_pulls = (
                self.resource.dummy
        )
        estimated_pulls = strict_pulls
        return strict_pulls, estimated_pulls

    def get_raw_result(
            self
    ) -> Tuple[Dict[model.EndfieldCharaWishModelState, np.ndarray], bool]:
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
        result: Dict[model.EndfieldCharaWishModelState, np.ndarray]
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
        result: Dict[model.EndfieldCharaWishModelState, np.ndarray],
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
    # Note: grid size is max_length + 1, so valid indices are 0 to max_length
    # stat + 1 might exceed max_length if stat >= max_length, so we need to clamp it
    grid = [[0 for _ in range(max_length + 1)] for _ in range(num_states)]
    for index, agg_stats in enumerate(agg_result.values()):
        for stat in agg_stats:
            grid[index][stat + 1] = 1

    return agg_result, grid, (num_states, max_length)


# ============================================================================
# Utility Functions
# ============================================================================

def init_endfield_user(user_name: str, passcode: str) -> EndfieldUser:
    """
    Initialize a Endfield user (placeholder for future authentication).

    Args:
        user_name: User name
        passcode: Passcode (not currently used)

    Returns:
        New EndfieldUser instance
    """
    return EndfieldUser(hash(user_name))


def show_graph():
    if platform.system() == 'Darwin':
        matplotlib.use('macosx')
    elif platform.system() == 'Linux':
        matplotlib.use('qtagg')
    else:
        matplotlib.use('TkAgg')

    # Initialize user
    user = EndfieldUser(1)

    # Set resources
    user.update_resource(consts.EndfieldResourceAction.ADJUST_DUMMY, 100)
    pull, pull_est = user.get_total_pull()

    # Set state
    user.set_state(
        0,
        0,
        True,
        [
            consts.EndfieldBannerType.CHARA,
            consts.EndfieldBannerType.CHARA,
            consts.EndfieldBannerType.CHARA,
            consts.EndfieldBannerType.CHARA,
            consts.EndfieldBannerType.CHARA,
        ]
    )

    # Trigger calculator
    user.trigger_calculator(force_cpu=False, use_dense=False)

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
