# RePlanTable AI
#
# Exact analytical IK + MuJoCo verification.
#
# AI assistance: ChatGPT was used to help implement the mathematical
# verification structure.
#
# The project concept, robot design, target behavior, testing, and final
# implementation decisions are the student's own work.

import sys
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from planner.kinematics import (  # noqa: E402
    solve_left_arm,
    solve_right_arm,
)


MODEL_XML = r"""
<mujoco model="replantable_exact_ik">

    <option
        timestep="0.002"
        gravity="0 0 -9.81"
    />


    <worldbody>


        <!-- Floor -->

        <geom
            name="floor"
            type="plane"
            size="5 5 0.1"
            pos="0 0 0"
            contype="0"
            conaffinity="0"
        />


        <!-- Table -->

        <body
            name="table"
            pos="0 0 0.9"
        >

            <geom
                name="table_top"
                type="box"
                size="1.4 0.8 0.05"
                contype="0"
                conaffinity="0"
            />

        </body>


        <!-- ================= LEFT ARM ================= -->

        <body
            name="left_base"
            pos="-1.1 0 1.02"
        >

            <joint
                name="left_joint_1"
                type="hinge"
                axis="0 1 0"
                range="-3.14159 3.14159"
            />

            <geom
                name="left_link_1"
                type="capsule"
                fromto="0 0 0 -0.35 0 0"
                size="0.07"
                contype="0"
                conaffinity="0"
            />


            <body
                name="left_link_2"
                pos="-0.35 0 0"
            >

                <joint
                    name="left_joint_2"
                    type="hinge"
                    axis="0 1 0"
                    range="-3.14159 3.14159"
                />

                <geom
                    name="left_link_2_geom"
                    type="capsule"
                    fromto="0 0 0 -0.35 0 0"
                    size="0.06"
                    contype="0"
                    conaffinity="0"
                />

                <site
                    name="left_gripper"
                    pos="-0.35 0 0"
                    size="0.05"
                />

            </body>

        </body>


        <!-- ================= RIGHT ARM ================= -->

        <body
            name="right_base"
            pos="1.1 0 1.02"
        >

            <joint
                name="right_joint_1"
                type="hinge"
                axis="0 1 0"
                range="-3.14159 3.14159"
            />

            <geom
                name="right_link_1"
                type="capsule"
                fromto="0 0 0 0.35 0 0"
                size="0.07"
                contype="0"
                conaffinity="0"
            />


            <body
                name="right_link_2"
                pos="0.35 0 0"
            >

                <joint
                    name="right_joint_2"
                    type="hinge"
                    axis="0 1 0"
                    range="-3.14159 3.14159"
                />

                <geom
                    name="right_link_2_geom"
                    type="capsule"
                    fromto="0 0 0 0.35 0 0"
                    size="0.06"
                    contype="0"
                    conaffinity="0"
                />

                <site
                    name="right_gripper"
                    pos="0.35 0 0"
                    size="0.05"
                />

            </body>

        </body>

    </worldbody>

</mujoco>
"""


def get_site_position(
    model,
    data,
    site_name,
):
    """Return a MuJoCo site's world position."""

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    if site_id < 0:

        raise ValueError(
            f"Site not found: {site_name}"
        )

    mujoco.mj_forward(
        model,
        data,
    )

    return data.site_xpos[
        site_id
    ].copy()


def set_joint_angle(
    model,
    data,
    joint_name,
    angle,
):
    """Set a named hinge joint angle."""

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if joint_id < 0:

        raise ValueError(
            f"Joint not found: {joint_name}"
        )

    qpos_address = (
        model.jnt_qposadr[joint_id]
    )

    data.qpos[
        qpos_address
    ] = angle


def verify_arm(
    model,
    data,
    side,
    target,
):
    """Solve IK, apply the result, and verify the actual MuJoCo position."""

    target = np.asarray(
        target,
        dtype=float,
    )


    if side == "left":

        q1, q2 = solve_left_arm(
            target_x=target[0],
            target_z=target[2],
        )

        joint_1 = "left_joint_1"
        joint_2 = "left_joint_2"

        site_name = "left_gripper"

    elif side == "right":

        q1, q2 = solve_right_arm(
            target_x=target[0],
            target_z=target[2],
        )

        joint_1 = "right_joint_1"
        joint_2 = "right_joint_2"

        site_name = "right_gripper"

    else:

        raise ValueError(
            "Unknown arm side."
        )


    print()
    print("=" * 60)

    print(
        f"{side.upper()} ARM → TARGET"
    )

    print("=" * 60)


    print(
        f"Target:"
        f" x={target[0]:.3f}"
        f" y={target[1]:.3f}"
        f" z={target[2]:.3f}"
    )


    print(
        "Calculated joint angles:"
    )

    print(
        f"  Joint 1: "
        f"{np.degrees(q1):.4f} degrees"
    )

    print(
        f"  Joint 2: "
        f"{np.degrees(q2):.4f} degrees"
    )


    # Apply exact calculated joint configuration.

    set_joint_angle(
        model,
        data,
        joint_1,
        q1,
    )

    set_joint_angle(
        model,
        data,
        joint_2,
        q2,
    )


    # Stop existing motion.

    data.qvel[:] = 0

    data.qacc[:] = 0

    mujoco.mj_forward(
        model,
        data,
    )


    actual = get_site_position(
        model,
        data,
        site_name,
    )


    error = float(
        np.linalg.norm(
            target - actual
        )
    )


    print()
    print(
        "MuJoCo actual gripper:"
    )

    print(
        f"  x={actual[0]:.4f}"
        f"  y={actual[1]:.4f}"
        f"  z={actual[2]:.4f}"
    )


    print()
    print(
        f"Position error: "
        f"{error:.6f} m"
    )


    tolerance = 0.01


    if error <= tolerance:

        print(
            "PASS: Target reached within 1 cm."
        )

        return True

    print(
        "FAIL: Target accuracy is outside 1 cm."
    )

    return False


def main():
    """Verify both arms using exact analytical IK."""

    model = mujoco.MjModel.from_xml_string(
        MODEL_XML
    )

    data = mujoco.MjData(
        model
    )


    print("=" * 60)

    print(
        "RePlanTable AI"
    )

    print(
        "Exact Analytical IK Verification"
    )

    print("=" * 60)


    left_pass = verify_arm(
        model=model,
        data=data,
        side="left",
        target=[
            -0.45,
            0.0,
            1.08,
        ],
    )


    right_pass = verify_arm(
        model=model,
        data=data,
        side="right",
        target=[
            0.45,
            0.0,
            1.20,
        ],
    )


    print()
    print("=" * 60)

    print(
        "FINAL RESULT"
    )

    print("=" * 60)


    if left_pass and right_pass:

        print(
            "PASS: Both arms reached their target positions."
        )

    else:

        print(
            "FAIL: One or more arms missed the target."
        )


if __name__ == "__main__":
    main()
