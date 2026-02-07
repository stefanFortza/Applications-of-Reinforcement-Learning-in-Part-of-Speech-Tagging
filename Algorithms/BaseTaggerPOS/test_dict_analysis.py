"""
Quick test to verify DICT analysis functionality works correctly.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
from Enviroment.BaseTaggerEnv.environment import PosCorrectionEnv
from Algorithms.BaseTaggerPOSUtils.dataset import (
    get_brown_as_universal,
    brown_to_training_data,
)
from Algorithms.BaseTaggerPOSUtils.rl_utils import (
    obs_to_discrete_state,
    analyze_dict_usage,
)


def test_dict_analysis():
    """Test DICT analysis with a simple mock agent."""
    print("Testing DICT Analysis Functionality\n")
    print("=" * 60)

    # Load small dataset
    brown_data = get_brown_as_universal()
    test_data = brown_to_training_data(brown_data[:10])

    # Create environment
    env = PosCorrectionEnv(test_data, mode="sequential")

    # Create a mock Q-table that always chooses DICT (action 2)
    mock_agent = {}

    # Simulate encountering some states and always choosing DICT
    for i in range(100):
        obs, info = env.reset()
        state = obs_to_discrete_state(obs)
        mock_agent[state] = np.array([0.0, 0.0, 1.0])  # Always DICT

    print("Mock Agent Created: Always chooses DICT action")
    print(f"Q-table size: {len(mock_agent)} states\n")

    # Run analysis
    print("Running DICT analysis...")
    dict_stats = analyze_dict_usage(
        agent=mock_agent,
        env=env,
        obs_to_state_fn=obs_to_discrete_state,
        num_episodes=10,
        is_dqn=False,
        is_policy=False,
    )

    # Print results
    print("\nResults:")
    print("-" * 60)
    print(f"Total DICT uses: {dict_stats['dict_total']}")
    print(f"Necessary (base was wrong): {dict_stats['dict_necessary']}")
    print(f"Unnecessary (base was correct): {dict_stats['dict_unnecessary']}")
    print(f"Necessity rate: {dict_stats['dict_necessity_rate']*100:.1f}%")
    print(f"Waste rate: {dict_stats['dict_waste_rate']*100:.1f}%")
    print(f"Action distribution: {dict_stats['action_counts']}")

    # Verify functionality
    print("\n" + "=" * 60)
    print("VERIFICATION:")

    total_actions = sum(dict_stats["action_counts"].values())
    dict_percentage = dict_stats["action_counts"]["DICT"] / total_actions * 100

    if dict_stats["dict_total"] > 0:
        print("✓ DICT action was used")
        print(f"  → Used in {dict_percentage:.1f}% of actions")
    else:
        print("✗ DICT action was never used (unexpected for mock agent)")
        return False

    if (
        dict_stats["dict_necessary"] + dict_stats["dict_unnecessary"]
        == dict_stats["dict_total"]
    ):
        print("✓ DICT counts are consistent")
    else:
        print("✗ DICT counts don't add up correctly")
        return False

    if 0 <= dict_stats["dict_necessity_rate"] <= 1:
        print("✓ Necessity rate is valid (0-1)")
    else:
        print("✗ Necessity rate is out of range")
        return False

    if 0 <= dict_stats["dict_waste_rate"] <= 1:
        print("✓ Waste rate is valid (0-1)")
    else:
        print("✗ Waste rate is out of range")
        return False

    if (
        abs(dict_stats["dict_necessity_rate"] + dict_stats["dict_waste_rate"] - 1.0)
        < 0.01
    ):
        print("✓ Necessity + Waste = 100%")
    else:
        print("✗ Necessity + Waste don't sum to 100%")
        return False

    print("\n" + "=" * 60)
    print("TEST PASSED: DICT analysis is working correctly!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_dict_analysis()
    sys.exit(0 if success else 1)
