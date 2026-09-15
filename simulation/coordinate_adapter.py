# RePlanTable AI
#
# Planner -> SO-101 coordinate adapter.
#
# AI assistance: OpenAI ChatGPT was used to help design
# the planner-to-robot coordinate mapping.

import numpy as np


# ============================================================
# PLANNER / ABSTRACT TASK SPACE
# ============================================================

PLANNER_X_MIN = -0.50
PLANNER_X_MAX = 0.50


# ============================================================
# SO-101 ROBOT WORKSPACE
# ============================================================

ROBOT_X_MIN = -0.10
ROBOT_X_MAX = 0.10

ROBOT_Y_MIN = -0.10
ROBOT_Y_MAX = 0.10

ROBOT_Z = 0.20


def planner_to_robot(target):
    """
    Convert high-level planner coordinates into
    SO-101 MuJoCo Cartesian coordinates.
    """

    target = np.asarray(
        target,
        dtype=float,
    )

    planner_x = float(target[0])
    planner_y = float(target[1])

    # Map planner X [-0.50, +0.50]
    # into robot X [-0.10, +0.10].
    normalized_x = (
        planner_x - PLANNER_X_MIN
    ) / (
        PLANNER_X_MAX - PLANNER_X_MIN
    )

    normalized_x = np.clip(
        normalized_x,
        0.0,
        1.0,
    )

    robot_x = (
        ROBOT_X_MIN
        + normalized_x
        * (
            ROBOT_X_MAX
            - ROBOT_X_MIN
        )
    )

    robot_y = np.clip(
        planner_y,
        ROBOT_Y_MIN,
        ROBOT_Y_MAX,
    )

    return np.array(
        [
            robot_x,
            robot_y,
            ROBOT_Z,
        ],
        dtype=float,
    )


def robot_to_planner(position):
    """
    Convert SO-101 MuJoCo coordinates back into
    abstract planner coordinates.
    """

    position = np.asarray(
        position,
        dtype=float,
    )

    robot_x = float(position[0])
    robot_y = float(position[1])

    normalized_x = (
        robot_x - ROBOT_X_MIN
    ) / (
        ROBOT_X_MAX - ROBOT_X_MIN
    )

    normalized_x = np.clip(
        normalized_x,
        0.0,
        1.0,
    )

    planner_x = (
        PLANNER_X_MIN
        + normalized_x
        * (
            PLANNER_X_MAX
            - PLANNER_X_MIN
        )
    )

    return np.array(
        [
            planner_x,
            robot_y,
            float(position[2]),
        ],
        dtype=float,
    )
