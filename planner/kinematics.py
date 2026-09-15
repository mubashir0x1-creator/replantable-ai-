# RePlanTable AI
#
# Analytical inverse kinematics for the simplified two-link arms.
#
# AI assistance: ChatGPT was used to help derive and implement the
# mathematical inverse-kinematics equations.
#
# The project concept, robot design, target behavior, testing, and final
# implementation decisions are the student's own work.

import math


LINK_1 = 0.35
LINK_2 = 0.35

LEFT_BASE_X = -1.10
RIGHT_BASE_X = 1.10

BASE_Z = 1.02


def normalize_angle(angle):
    """Normalize an angle to the range [-pi, pi]."""

    while angle > math.pi:
        angle -= 2.0 * math.pi

    while angle < -math.pi:
        angle += 2.0 * math.pi

    return angle


def solve_standard_two_link(
    x,
    z,
    elbow_sign=1.0,
):
    """
    Solve a standard two-link planar arm.

    The standard mathematical arm points along +X when
    both joint angles are zero.
    """

    distance_squared = (
        x * x
        + z * z
    )

    distance = math.sqrt(
        distance_squared
    )

    maximum_reach = (
        LINK_1
        + LINK_2
    )

    minimum_reach = abs(
        LINK_1
        - LINK_2
    )

    if distance > maximum_reach + 1e-9:

        raise ValueError(
            f"Target is outside maximum reach: "
            f"{distance:.4f} m > {maximum_reach:.4f} m."
        )

    if distance < minimum_reach - 1e-9:

        raise ValueError(
            f"Target is inside minimum reach: "
            f"{distance:.4f} m < {minimum_reach:.4f} m."
        )

    cos_q2 = (
        distance_squared
        - LINK_1 * LINK_1
        - LINK_2 * LINK_2
    ) / (
        2.0
        * LINK_1
        * LINK_2
    )

    cos_q2 = max(
        -1.0,
        min(1.0, cos_q2),
    )

    q2 = (
        elbow_sign
        * math.acos(cos_q2)
    )

    q1 = (
        math.atan2(z, x)
        - math.atan2(
            LINK_2 * math.sin(q2),
            LINK_1
            + LINK_2 * math.cos(q2),
        )
    )

    return (
        normalize_angle(q1),
        normalize_angle(q2),
    )


def solve_left_arm(
    target_x,
    target_z,
):
    """
    Solve the left MuJoCo arm.

    At zero angles, the left arm points toward negative X.
    """

    # MuJoCo left-arm geometry:
    #
    # x = base_x - standard_x
    # z = base_z + standard_z
    #
    # Therefore:
    #
    # standard_x = base_x - target_x
    # standard_z = target_z - base_z

    standard_x = (
        LEFT_BASE_X
        - target_x
    )

    standard_z = (
        target_z
        - BASE_Z
    )

    q1, q2 = solve_standard_two_link(
        x=standard_x,
        z=standard_z,
        elbow_sign=1.0,
    )

    # IMPORTANT:
    # Unlike the previous version, do NOT add pi here.
    #
    # The MuJoCo joint angles already describe the rotation of the
    # left links from their negative-X zero pose.

    return (
        q1,
        q2,
    )


def solve_right_arm(
    target_x,
    target_z,
):
    """
    Solve the right MuJoCo arm.

    At zero angles, the right arm points toward positive X.
    """

    # MuJoCo right-arm geometry:
    #
    # x = base_x + standard_x
    # z = base_z - standard_z
    #
    # Therefore:
    #
    # standard_x = target_x - base_x
    # standard_z = base_z - target_z

    standard_x = (
        target_x
        - RIGHT_BASE_X
    )

    standard_z = (
        BASE_Z
        - target_z
    )

    q1, q2 = solve_standard_two_link(
        x=standard_x,
        z=standard_z,
        elbow_sign=-1.0,
    )

    return (
        q1,
        q2,
    )
