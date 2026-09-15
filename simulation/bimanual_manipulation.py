# RePlanTable AI
#
# Bimanual pick-and-place demonstration in MuJoCo.
#
# AI assistance: ChatGPT was used to help draft the initial MuJoCo
# manipulation controller and finger-geometry implementation.
#
# The project concept, manipulation workflow, target placement,
# verification logic, and final implementation decisions are the
# student's own work.


import sys
from pathlib import Path

import mujoco
import numpy as np


# ------------------------------------------------------------
# Project imports
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# MuJoCo model
# ------------------------------------------------------------

MODEL_XML = r"""
<mujoco model="replantable_bimanual">

    <!--
        This stage uses kinematic control.

        Gravity and collisions are disabled so that manipulation
        logic can be verified deterministically before introducing
        full rigid-body dynamics.
    -->

    <option
        timestep="0.002"
        gravity="0 0 0"
    />


    <worldbody>

        <!-- ================================================= -->
        <!-- FLOOR                                             -->
        <!-- ================================================= -->

        <geom
            name="floor"
            type="plane"
            size="5 5 0.1"
            pos="0 0 0"
            contype="0"
            conaffinity="0"
        />


        <!-- ================================================= -->
        <!-- TABLE                                             -->
        <!-- ================================================= -->

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


        <!-- ================================================= -->
        <!-- LEFT ARM                                           -->
        <!-- ================================================= -->

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
                name="left_link_1_geom"
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


                <!-- End-effector center -->

                <site
                    name="left_gripper"
                    pos="-0.35 0 0"
                    size="0.035"
                />


                <!-- Left gripper fingers -->

                <geom
                    name="left_finger_a"
                    type="box"
                    size="0.055 0.012 0.035"
                    pos="-0.385 0.06 0"
                    contype="0"
                    conaffinity="0"
                />

                <geom
                    name="left_finger_b"
                    type="box"
                    size="0.055 0.012 0.035"
                    pos="-0.385 -0.06 0"
                    contype="0"
                    conaffinity="0"
                />

            </body>

        </body>


        <!-- ================================================= -->
        <!-- RIGHT ARM                                          -->
        <!-- ================================================= -->

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
                name="right_link_1_geom"
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


                <!-- End-effector center -->

                <site
                    name="right_gripper"
                    pos="0.35 0 0"
                    size="0.035"
                />


                <!-- Right gripper fingers -->

                <geom
                    name="right_finger_a"
                    type="box"
                    size="0.055 0.012 0.035"
                    pos="0.385 0.06 0"
                    contype="0"
                    conaffinity="0"
                />

                <geom
                    name="right_finger_b"
                    type="box"
                    size="0.055 0.012 0.035"
                    pos="0.385 -0.06 0"
                    contype="0"
                    conaffinity="0"
                />

            </body>

        </body>


        <!-- ================================================= -->
        <!-- PLATE                                              -->
        <!-- ================================================= -->

        <body
            name="plate"
            pos="-0.45 0 1.08"
        >

            <freejoint
                name="plate_freejoint"
            />

            <geom
                name="plate_geom"
                type="cylinder"
                size="0.20 0.025"
                mass="0.2"
                contype="0"
                conaffinity="0"
            />

        </body>


        <!-- ================================================= -->
        <!-- CUP                                                -->
        <!-- ================================================= -->

        <body
            name="cup"
            pos="0.45 0 1.20"
        >

            <freejoint
                name="cup_freejoint"
            />

            <geom
                name="cup_geom"
                type="cylinder"
                size="0.08 0.12"
                mass="0.15"
                contype="0"
                conaffinity="0"
            />

        </body>

    </worldbody>

</mujoco>
"""


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def get_joint_qpos_address(
    model,
    joint_name,
):
    """Return the qpos address of a named joint."""

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if joint_id < 0:
        raise ValueError(
            f"Joint not found: {joint_name}"
        )

    return model.jnt_qposadr[joint_id]


def get_body_id(
    model,
    body_name,
):
    """Return the body ID for a named body."""

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )

    if body_id < 0:
        raise ValueError(
            f"Body not found: {body_name}"
        )

    return body_id


def get_site_id(
    model,
    site_name,
):
    """Return the site ID."""

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    if site_id < 0:
        raise ValueError(
            f"Site not found: {site_name}"
        )

    return site_id


def get_geom_id(
    model,
    geom_name,
):
    """Return the geom ID."""

    geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_name,
    )

    if geom_id < 0:
        raise ValueError(
            f"Geom not found: {geom_name}"
        )

    return geom_id


def get_site_position(
    model,
    data,
    site_name,
):
    """Return a site's world position."""

    site_id = get_site_id(
        model,
        site_name,
    )

    mujoco.mj_forward(
        model,
        data,
    )

    return data.site_xpos[
        site_id
    ].copy()


def get_body_position(
    model,
    data,
    body_name,
):
    """Return a body's world position."""

    body_id = get_body_id(
        model,
        body_name,
    )

    mujoco.mj_forward(
        model,
        data,
    )

    return data.xpos[
        body_id
    ].copy()


def distance(
    point_a,
    point_b,
):
    """Return Euclidean distance."""

    return float(
        np.linalg.norm(
            point_a - point_b
        )
    )


def shortest_angle_delta(
    current,
    target,
):
    """Return the shortest angular difference."""

    return (
        target
        - current
        + np.pi
    ) % (
        2 * np.pi
    ) - np.pi


# ------------------------------------------------------------
# Joint control
# ------------------------------------------------------------

def get_arm_joint_positions(
    model,
    data,
    joint_names,
):
    """Read two arm joint positions."""

    return np.array(
        [
            data.qpos[
                get_joint_qpos_address(
                    model,
                    joint_name,
                )
            ]
            for joint_name in joint_names
        ],
        dtype=float,
    )


def set_arm_joint_positions(
    model,
    data,
    joint_names,
    values,
):
    """Set two arm joint positions."""

    for joint_name, value in zip(
        joint_names,
        values,
    ):

        address = get_joint_qpos_address(
            model,
            joint_name,
        )

        data.qpos[address] = value


def move_arm_pair(
    model,
    data,
    left_target,
    right_target,
    grasped_objects,
    steps=120,
):
    """
    Move both arms simultaneously.

    Objects currently held by a gripper follow that gripper.
    """

    left_angles = solve_left_arm(
        target_x=left_target[0],
        target_z=left_target[2],
    )

    right_angles = solve_right_arm(
        target_x=right_target[0],
        target_z=right_target[2],
    )


    left_joints = [
        "left_joint_1",
        "left_joint_2",
    ]

    right_joints = [
        "right_joint_1",
        "right_joint_2",
    ]


    left_start = get_arm_joint_positions(
        model,
        data,
        left_joints,
    )

    right_start = get_arm_joint_positions(
        model,
        data,
        right_joints,
    )


    left_delta = np.array(
        [
            shortest_angle_delta(
                left_start[index],
                left_angles[index],
            )
            for index in range(2)
        ]
    )

    right_delta = np.array(
        [
            shortest_angle_delta(
                right_start[index],
                right_angles[index],
            )
            for index in range(2)
        ]
    )


    for step in range(
        1,
        steps + 1,
    ):

        progress = step / steps


        left_current = (
            left_start
            + progress * left_delta
        )

        right_current = (
            right_start
            + progress * right_delta
        )


        set_arm_joint_positions(
            model,
            data,
            left_joints,
            left_current,
        )

        set_arm_joint_positions(
            model,
            data,
            right_joints,
            right_current,
        )


        mujoco.mj_forward(
            model,
            data,
        )


        update_grasped_objects(
            model,
            data,
            grasped_objects,
        )


        mujoco.mj_forward(
            model,
            data,
        )


# ------------------------------------------------------------
# Gripper
# ------------------------------------------------------------

def set_gripper_width(
    model,
    arm,
    width,
):
    """
    Change the visual gripper opening.

    Width is the distance from the centerline to each finger.
    """

    if arm == "left":

        finger_a = "left_finger_a"
        finger_b = "left_finger_b"

    elif arm == "right":

        finger_a = "right_finger_a"
        finger_b = "right_finger_b"

    else:

        raise ValueError(
            "arm must be 'left' or 'right'."
        )


    geom_a = get_geom_id(
        model,
        finger_a,
    )

    geom_b = get_geom_id(
        model,
        finger_b,
    )


    model.geom_pos[
        geom_a,
        1
    ] = width

    model.geom_pos[
        geom_b,
        1
    ] = -width


def open_both_grippers(
    model,
):
    """Open both visual grippers."""

    set_gripper_width(
        model,
        "left",
        0.06,
    )

    set_gripper_width(
        model,
        "right",
        0.06,
    )


def close_both_grippers(
    model,
):
    """Close both visual grippers."""

    set_gripper_width(
        model,
        "left",
        0.025,
    )

    set_gripper_width(
        model,
        "right",
        0.025,
    )


# ------------------------------------------------------------
# Object manipulation
# ------------------------------------------------------------

def get_freejoint_position(
    model,
    data,
    joint_name,
):
    """Return the position stored by a free joint."""

    address = get_joint_qpos_address(
        model,
        joint_name,
    )

    return data.qpos[
        address:address + 3
    ].copy()


def set_freejoint_position(
    model,
    data,
    joint_name,
    position,
):
    """Set the position of a free joint."""

    address = get_joint_qpos_address(
        model,
        joint_name,
    )

    data.qpos[
        address:address + 3
    ] = np.asarray(
        position,
        dtype=float,
    )


def preserve_object_orientation(
    model,
    data,
    joint_name,
):
    """Set a free joint orientation to identity."""

    address = get_joint_qpos_address(
        model,
        joint_name,
    )

    data.qpos[
        address + 3:address + 7
    ] = np.array(
        [
            1.0,
            0.0,
            0.0,
            0.0,
        ]
    )


def grasp_object(
    model,
    data,
    arm,
    object_name,
    grasped_objects,
):
    """
    Attach an object to a gripper.

    The object retains its offset from the gripper center.
    """

    if arm == "left":

        site_name = "left_gripper"

    elif arm == "right":

        site_name = "right_gripper"

    else:

        raise ValueError(
            "Unknown arm."
        )


    object_body_position = (
        get_body_position(
            model,
            data,
            object_name,
        )
    )


    gripper_position = (
        get_site_position(
            model,
            data,
            site_name,
        )
    )


    gap = distance(
        object_body_position,
        gripper_position,
    )


    print(
        f"{arm.upper()} grasp distance: "
        f"{gap:.4f} m"
    )


    if gap > 0.08:

        raise RuntimeError(
            f"{object_name} is too far "
            f"from the {arm} gripper to grasp."
        )


    if object_name == "plate":

        joint_name = "plate_freejoint"

    elif object_name == "cup":

        joint_name = "cup_freejoint"

    else:

        raise ValueError(
            f"Unknown object: {object_name}"
        )


    grasped_objects[object_name] = {
        "arm": arm,
        "site_name": site_name,
        "joint_name": joint_name,
        "offset": (
            object_body_position
            - gripper_position
        ),
    }


    print(
        f"GRASPED: {object_name} "
        f"with {arm} arm"
    )


def update_grasped_objects(
    model,
    data,
    grasped_objects,
):
    """Move grasped objects with their grippers."""

    for object_name, state in (
        grasped_objects.items()
    ):

        gripper_position = (
            get_site_position(
                model,
                data,
                state["site_name"],
            )
        )


        new_position = (
            gripper_position
            + state["offset"]
        )


        set_freejoint_position(
            model,
            data,
            state["joint_name"],
            new_position,
        )


        preserve_object_orientation(
            model,
            data,
            state["joint_name"],
        )


def release_object(
    model,
    data,
    object_name,
    grasped_objects,
):
    """Release an object from its gripper."""

    if object_name not in grasped_objects:

        raise RuntimeError(
            f"{object_name} is not currently grasped."
        )


    final_position = get_body_position(
        model,
        data,
        object_name,
    )


    print(
        f"RELEASED: {object_name} "
        f"at "
        f"x={final_position[0]:.3f}, "
        f"y={final_position[1]:.3f}, "
        f"z={final_position[2]:.3f}"
    )


    del grasped_objects[
        object_name
    ]


# ------------------------------------------------------------
# Verification
# ------------------------------------------------------------

def verify_object(
    model,
    data,
    object_name,
    expected,
):
    """Verify an object's final position."""

    actual = get_body_position(
        model,
        data,
        object_name,
    )


    error = distance(
        actual,
        np.asarray(
            expected,
            dtype=float,
        ),
    )


    print(
        f"{object_name.upper()} FINAL POSITION"
    )


    print(
        f"Expected:"
        f" x={expected[0]:.3f}"
        f" y={expected[1]:.3f}"
        f" z={expected[2]:.3f}"
    )


    print(
        f"Actual:"
        f" x={actual[0]:.3f}"
        f" y={actual[1]:.3f}"
        f" z={actual[2]:.3f}"
    )


    print(
        f"Position error: "
        f"{error:.4f} m"
    )


    passed = (
        error <= 0.05
    )


    if passed:

        print(
            "PASS: Object reached its destination."
        )

    else:

        print(
            "FAIL: Object missed its destination."
        )


    return passed


# ------------------------------------------------------------
# Main manipulation workflow
# ------------------------------------------------------------

def execute_planned_task(plan, force_failure=False):
    """
    Execute a planner-generated pick-and-place plan
    using the existing MuJoCo manipulation system.
    """

    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)

    open_both_grippers(model)

    mujoco.mj_forward(model, data)

    print()
    print("=" * 64)
    print("EXECUTING PLANNER TASK")
    print("=" * 64)

    # --------------------------------------------------------
    # Find pick and place targets from planner
    # --------------------------------------------------------

    left_pick = None
    right_pick = None
    left_place = None
    right_place = None

    for action in plan:

        if action.action == "pick":

            if action.object_name == "plate":
                left_pick = np.array(
                    action.target,
                    dtype=float,
                )

            elif action.object_name == "cup":
                right_pick = np.array(
                    action.target,
                    dtype=float,
                )

        elif action.action == "place":

            if action.object_name == "plate":
                left_place = np.array(
                    action.target,
                    dtype=float,
                )

            elif action.object_name == "cup":
                right_place = np.array(
                    action.target,
                    dtype=float,
                )

    # --------------------------------------------------------
    # Make sure the planner produced complete targets
    # --------------------------------------------------------

    if left_pick is None:
        raise RuntimeError("Planner did not provide plate pick target.")

    if right_pick is None:
        raise RuntimeError("Planner did not provide cup pick target.")

    if left_place is None:
        raise RuntimeError("Planner did not provide plate place target.")

    if right_place is None:
        raise RuntimeError("Planner did not provide cup place target.")

    # --------------------------------------------------------
    # STEP 1 — Move to planner targets
    # --------------------------------------------------------

    print()
    print("STEP 1 — PLANNER TARGETS")

    move_arm_pair(
        model=model,
        data=data,
        left_target=left_pick,
        right_target=right_pick,
        grasped_objects={},
        steps=120,
    )

    mujoco.mj_forward(model, data)

    # --------------------------------------------------------
    # STEP 2 — Grasp
    # --------------------------------------------------------

    print()
    print("STEP 2 — GRASP")

    close_both_grippers(model)

    grasped_objects = {}

    grasp_object(
        model=model,
        data=data,
        arm="left",
        object_name="plate",
        grasped_objects=grasped_objects,
    )

    grasp_object(
        model=model,
        data=data,
        arm="right",
        object_name="cup",
        grasped_objects=grasped_objects,
    )

    print(
        f"Grasped objects: {len(grasped_objects)}"
    )

    # --------------------------------------------------------
    # STEP 3 — Execute planner place targets
    # --------------------------------------------------------

    print()
    print("STEP 3 — CARRY TO PLANNER TARGETS")

    move_arm_pair(
        model=model,
        data=data,
        left_target=left_place,
        right_target=right_place,
        grasped_objects=grasped_objects,
        steps=160,
    )

    # --------------------------------------------------------
    # STEP 4 — Release
    # --------------------------------------------------------

    print()
    print("STEP 4 — RELEASE")

    release_object(
        model=model,
        data=data,
        object_name="plate",
        grasped_objects=grasped_objects,
    )

    release_object(
        model=model,
        data=data,
        object_name="cup",
        grasped_objects=grasped_objects,
    )

    open_both_grippers(model)

    mujoco.mj_forward(model, data)

    # --------------------------------------------------------
    # OPTIONAL FAILURE INJECTION
    # --------------------------------------------------------
    #
    # Used only to test the automatic failure/replanning
    # pipeline. Normal execution keeps this False.
    #

    if force_failure:

        print()
        print("=" * 64)
        print("FAILURE INJECTION")
        print("=" * 64)

        print(
            "Simulating a verification failure for the cup."
        )

        cup_actual = get_body_position(
            model,
            data,
            "cup",
        )

        failed_position = (
            cup_actual
            + np.array(
                [0.20, 0.0, 0.0],
                dtype=float,
            )
        )

        set_freejoint_position(
            model,
            data,
            "cup_freejoint",
            failed_position,
        )

        mujoco.mj_forward(
            model,
            data,
        )

    # --------------------------------------------------------
    # STEP 5 — Verify
    # --------------------------------------------------------

    print()
    print("STEP 5 — VERIFY")

    plate_pass = verify_object(
        model=model,
        data=data,
        object_name="plate",
        expected=left_place,
    )

    print()

    cup_pass = verify_object(
        model=model,
        data=data,
        object_name="cup",
        expected=right_place,
    )

        # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("=" * 64)
    print("PLANNER EXECUTION RESULT")
    print("=" * 64)

    failed_objects = []

    if not plate_pass:
        failed_objects.append("plate")

    if not cup_pass:
        failed_objects.append("cup")

    if not failed_objects:

        print(
            "PASS: Planner task executed successfully."
        )

        return {
            "success": True,
            "failed_objects": [],
        }

    print(
        "FAIL: Planner task verification failed."
    )

    print()
    print("FAILED OBJECTS:")

    for object_name in failed_objects:
        print(
            f"- {object_name}"
        )

    return {
        "success": False,
        "failed_objects": failed_objects,
    }

def main():
    """Run the full bimanual manipulation demonstration."""

    model = mujoco.MjModel.from_xml_string(
        MODEL_XML
    )

    data = mujoco.MjData(
        model
    )


    print(
        "=" * 64
    )

    print(
        "RePlanTable AI"
    )

    print(
        "Bimanual Pick-and-Place Test"
    )

    print(
        "=" * 64
    )


    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    open_both_grippers(
        model
    )


    mujoco.mj_forward(
        model,
        data,
    )


    print()

    print(
        "INITIAL OBJECTS"
    )


    plate_initial = get_body_position(
        model,
        data,
        "plate",
    )

    cup_initial = get_body_position(
        model,
        data,
        "cup",
    )


    print(
        f"Plate:"
        f" x={plate_initial[0]:.3f}"
        f" y={plate_initial[1]:.3f}"
        f" z={plate_initial[2]:.3f}"
    )


    print(
        f"Cup:"
        f" x={cup_initial[0]:.3f}"
        f" y={cup_initial[1]:.3f}"
        f" z={cup_initial[2]:.3f}"
    )


    # --------------------------------------------------------
    # Pick targets
    # --------------------------------------------------------

    left_pick = np.array(
        [
            -0.45,
            0.0,
            1.08,
        ],
        dtype=float,
    )


    right_pick = np.array(
        [
            0.45,
            0.0,
            1.20,
        ],
        dtype=float,
    )


    # --------------------------------------------------------
    # STEP 1 — Move to objects
    # --------------------------------------------------------

    print()

    print(
        "=" * 64
    )

    print(
        "STEP 1 — MOVE TO OBJECTS"
    )

    print(
        "=" * 64
    )


    move_arm_pair(
        model=model,
        data=data,
        left_target=left_pick,
        right_target=right_pick,
        grasped_objects={},
        steps=120,
    )


    mujoco.mj_forward(
        model,
        data,
    )


    left_gripper = get_site_position(
        model,
        data,
        "left_gripper",
    )


    right_gripper = get_site_position(
        model,
        data,
        "right_gripper",
    )


    print(
        f"Left gripper:"
        f" x={left_gripper[0]:.3f}"
        f" y={left_gripper[1]:.3f}"
        f" z={left_gripper[2]:.3f}"
    )


    print(
        f"Right gripper:"
        f" x={right_gripper[0]:.3f}"
        f" y={right_gripper[1]:.3f}"
        f" z={right_gripper[2]:.3f}"
    )


    # --------------------------------------------------------
    # Verify pickup position
    # --------------------------------------------------------

    plate_gap = distance(
        left_gripper,
        plate_initial,
    )


    cup_gap = distance(
        right_gripper,
        cup_initial,
    )


    print()

    print(
        f"Plate pickup gap: "
        f"{plate_gap:.4f} m"
    )


    print(
        f"Cup pickup gap:   "
        f"{cup_gap:.4f} m"
    )


    if plate_gap > 0.08:

        raise RuntimeError(
            "Left gripper did not reach the plate."
        )


    if cup_gap > 0.08:

        raise RuntimeError(
            "Right gripper did not reach the cup."
        )


    # --------------------------------------------------------
    # STEP 2 — Grasp
    # --------------------------------------------------------

    print()

    print(
        "=" * 64
    )

    print(
        "STEP 2 — GRASP"
    )

    print(
        "=" * 64
    )


    close_both_grippers(
        model
    )


    grasped_objects = {}


    grasp_object(
        model=model,
        data=data,
        arm="left",
        object_name="plate",
        grasped_objects=grasped_objects,
    )


    grasp_object(
        model=model,
        data=data,
        arm="right",
        object_name="cup",
        grasped_objects=grasped_objects,
    )


    print(
        f"Grasped objects: "
        f"{len(grasped_objects)}"
    )


    # --------------------------------------------------------
    # Place targets
    # --------------------------------------------------------

    left_place = np.array(
        [
            -0.42,
            0.0,
            1.08,
        ],
        dtype=float,
    )


    right_place = np.array(
        [
            0.44,
            0.0,
            1.20,
        ],
        dtype=float,
    )


    # --------------------------------------------------------
    # STEP 3 — Bimanual carry
    # --------------------------------------------------------

    print()

    print(
        "=" * 64
    )

    print(
        "STEP 3 — Bimanual carry"
    )

    print(
        "=" * 64
    )


    move_arm_pair(
        model=model,
        data=data,
        left_target=left_place,
        right_target=right_place,
        grasped_objects=grasped_objects,
        steps=160,
    )


    # --------------------------------------------------------
    # STEP 4 — Release
    # --------------------------------------------------------

    print()

    print(
        "=" * 64
    )

    print(
        "STEP 4 — RELEASE"
    )

    print(
        "=" * 64
    )


    release_object(
        model=model,
        data=data,
        object_name="plate",
        grasped_objects=grasped_objects,
    )


    release_object(
        model=model,
        data=data,
        object_name="cup",
        grasped_objects=grasped_objects,
    )


    open_both_grippers(
        model
    )


    mujoco.mj_forward(
        model,
        data,
    )


    # --------------------------------------------------------
    # STEP 5 — Verify
    # --------------------------------------------------------

    print()

    print(
        "=" * 64
    )

    print(
        "STEP 5 — VERIFY"
    )

    print(
        "=" * 64
    )


    plate_pass = verify_object(
        model=model,
        data=data,
        object_name="plate",
        expected=left_place,
    )


    print()


    cup_pass = verify_object(
        model=model,
        data=data,
        object_name="cup",
        expected=right_place,
    )


    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()

    print(
        "=" * 64
    )

    print(
        "FINAL RESULT"
    )

    print(
        "=" * 64
    )


    if (
        plate_pass
        and cup_pass
        and len(grasped_objects) == 0
    ):

        print(
            "PASS: Bimanual pick-and-place completed successfully."
        )

        print(
            "PASS: Both objects were grasped, carried, "
            "placed, and released."
        )

    else:

        print(
            "FAIL: Manipulation workflow needs adjustment."
        )


# ------------------------------------------------------------
# Program entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    main()
