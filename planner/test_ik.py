# RePlanTable AI
#
# Test the inverse-kinematics solver before connecting it to MuJoCo.

import math

from kinematics import (
    solve_left_arm,
    solve_right_arm,
)


def degrees(radians):
    """Convert radians to degrees."""

    return math.degrees(radians)


def main():

    print("=" * 60)
    print("RePlanTable AI - Inverse Kinematics Test")
    print("=" * 60)


    # Test target for the left arm.

    left_target = (
        -0.45,
        0.18,
    )

    left_q1, left_q2 = solve_left_arm(
        target_x=left_target[0],
        target_z=left_target[1],
    )

    print("\nLEFT ARM")

    print(
        f"Target: x={left_target[0]:.2f}, "
        f"z={left_target[1]:.2f}"
    )

    print(
        f"Joint 1: {degrees(left_q1):.2f} degrees"
    )

    print(
        f"Joint 2: {degrees(left_q2):.2f} degrees"
    )


    # Test target for the right arm.

    right_target = (
        0.45,
        0.18,
    )

    right_q1, right_q2 = solve_right_arm(
        target_x=right_target[0],
        target_z=right_target[1],
    )

    print("\nRIGHT ARM")

    print(
        f"Target: x={right_target[0]:.2f}, "
        f"z={right_target[1]:.2f}"
    )

    print(
        f"Joint 1: {degrees(right_q1):.2f} degrees"
    )

    print(
        f"Joint 2: {degrees(right_q2):.2f} degrees"
    )


    print("\nIK solver test completed.")


if __name__ == "__main__":
    main()
