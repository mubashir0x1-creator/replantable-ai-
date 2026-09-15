import numpy as np
from pathlib import Path
import mujoco


# ============================================================
# MODEL
# ============================================================

MODEL_XML = str(Path(__file__).resolve().parent / "dual_so101.xml")

model = mujoco.MjModel.from_xml_path(MODEL_XML)


# ============================================================
# JOINT NAMES
# ============================================================

LEFT_JOINT_NAMES = [
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow_flex",
    "left_wrist_flex",
    "left_wrist_roll",
]

RIGHT_JOINT_NAMES = [
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
    "right_wrist_roll",
]


LEFT_GRIPPER_NAME = "left_gripper"
RIGHT_GRIPPER_NAME = "right_gripper"

LEFT_SITE_NAME = "left_gripperframe"
RIGHT_SITE_NAME = "right_gripperframe"


# ============================================================
# ID HELPERS
# ============================================================

def joint_id(name):
    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name
    )

    if jid < 0:
        raise ValueError(f"Joint not found: {name}")

    return model.jnt_qposadr[jid]


def actuator_id(name):
    aid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name
    )

    if aid < 0:
        raise ValueError(f"Actuator not found: {name}")

    return aid


def site_id(name):
    sid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        name
    )

    if sid < 0:
        raise ValueError(f"Site not found: {name}")

    return sid


# ============================================================
# JOINT / ACTUATOR IDS
# ============================================================

LEFT_JOINT_IDS = [
    joint_id(name)
    for name in LEFT_JOINT_NAMES
]

RIGHT_JOINT_IDS = [
    joint_id(name)
    for name in RIGHT_JOINT_NAMES
]

LEFT_GRIPPER_ID = actuator_id(
    LEFT_GRIPPER_NAME
)

RIGHT_GRIPPER_ID = actuator_id(
    RIGHT_GRIPPER_NAME
)

LEFT_SITE_ID = site_id(
    LEFT_SITE_NAME
)

RIGHT_SITE_ID = site_id(
    RIGHT_SITE_NAME
)


# ============================================================
# SIMULATION
# ============================================================

def create_simulation():

    data = mujoco.MjData(model)

    # Open both grippers
    data.ctrl[LEFT_GRIPPER_ID] = 0.5
    data.ctrl[RIGHT_GRIPPER_ID] = 0.5

    mujoco.mj_forward(model, data)

    return data


# ============================================================
# POSITION HELPERS
# ============================================================

def get_gripper_position(data, arm):

    if arm == "left":
        return data.site_xpos[LEFT_SITE_ID].copy()

    if arm == "right":
        return data.site_xpos[RIGHT_SITE_ID].copy()

    raise ValueError(f"Unknown arm: {arm}")


def get_site_position(data, name):

    sid = site_id(name)

    return data.site_xpos[sid].copy()


def get_body_position(data, name):

    bid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        name
    )

    if bid < 0:
        raise ValueError(f"Body not found: {name}")

    return data.xpos[bid].copy()


# ============================================================
# GRIPPER CONTROL
# ============================================================

def open_gripper(data, arm):

    if arm == "left":

        data.ctrl[
            LEFT_GRIPPER_ID
        ] = 0.5

        print("LEFT gripper OPEN")

    elif arm == "right":

        data.ctrl[
            RIGHT_GRIPPER_ID
        ] = 0.5

        print("RIGHT gripper OPEN")

    else:
        raise ValueError(f"Unknown arm: {arm}")


def close_gripper(data, arm):

    if arm == "left":

        data.ctrl[
            LEFT_GRIPPER_ID
        ] = 1.0

        print("LEFT gripper CLOSED")

    elif arm == "right":

        data.ctrl[
            RIGHT_GRIPPER_ID
        ] = 1.0

        print("RIGHT gripper CLOSED")

    else:
        raise ValueError(f"Unknown arm: {arm}")


def open_both_grippers(data):

    open_gripper(data, "left")
    open_gripper(data, "right")


def close_both_grippers(data):

    close_gripper(data, "left")
    close_gripper(data, "right")


# ============================================================
# NUMERICAL IK
# ============================================================

def solve_ik(
    data,
    target,
    arm,
    iterations=600,
    tolerance=0.003
):

    target = np.asarray(
        target,
        dtype=float
    )

    if arm == "left":

        joint_ids = LEFT_JOINT_IDS
        sid = LEFT_SITE_ID

    elif arm == "right":

        joint_ids = RIGHT_JOINT_IDS
        sid = RIGHT_SITE_ID

    else:
        raise ValueError(
            f"Unknown arm: {arm}"
        )

    for _ in range(iterations):

        mujoco.mj_forward(
            model,
            data
        )

        current = data.site_xpos[
            sid
        ].copy()

        error = target - current

        error_norm = np.linalg.norm(
            error
        )

        if error_norm < tolerance:
            break

        jacp = np.zeros(
            (3, model.nv)
        )

        jacr = np.zeros(
            (3, model.nv)
        )

        mujoco.mj_jacSite(
            model,
            data,
            jacp,
            jacr,
            sid
        )

        J = jacp[
            :,
            joint_ids
        ]

        damping = 0.01

        dq = (
            J.T
            @ np.linalg.solve(
                J @ J.T
                + damping
                * np.eye(3),
                error
            )
        )

        for i, joint in enumerate(
            joint_ids
        ):

            data.qpos[joint] += dq[i]

            # Respect joint limits
            jid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                (
                    LEFT_JOINT_NAMES[i]
                    if arm == "left"
                    else RIGHT_JOINT_NAMES[i]
                )
            )

            if model.jnt_limited[jid]:

                low = model.jnt_range[
                    jid,
                    0
                ]

                high = model.jnt_range[
                    jid,
                    1
                ]

                data.qpos[joint] = np.clip(
                    data.qpos[joint],
                    low,
                    high
                )

    mujoco.mj_forward(
        model,
        data
    )

    final_position = data.site_xpos[
        sid
    ].copy()

    final_error = np.linalg.norm(
        target - final_position
    )

    return final_position, final_error


# ============================================================
# LOGICAL GRASP STATE
# ============================================================

GRASPED_OBJECTS = {}


# ============================================================
# UPDATE GRASPED OBJECTS
# ============================================================

def update_grasped_objects(data):

    if not GRASPED_OBJECTS:
        return

    for object_name, info in list(
        GRASPED_OBJECTS.items()
    ):

        arm = info["arm"]

        gripper_position = (
            get_gripper_position(
                data,
                arm
            )
        )

        # Keep the original grasp offset in X/Y.
        # For Z, keep the object's original height so the
        # logical grasp does not push the object into the table.
        offset = info["offset"]

        new_position = (
            gripper_position + offset
        )

        joint_name = (
            f"{object_name}_free"
        )

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name
        )

        if jid < 0:
            continue

        qpos_address = (
            model.jnt_qposadr[jid]
        )

        data.qpos[
            qpos_address:
            qpos_address + 3
        ] = new_position

    mujoco.mj_forward(
        model,
        data
    )


# ============================================================
# ARM MOVEMENT
# ============================================================

def move_arm(
    data,
    arm,
    target,
    steps=1200
):

    target = np.asarray(
        target,
        dtype=float
    )

    print(
        f"\nMoving {arm} arm to:"
        f" x={target[0]:.3f}"
        f" y={target[1]:.3f}"
        f" z={target[2]:.3f}"
    )

    # --------------------------------------------------------
    # Select joints
    # --------------------------------------------------------

    if arm == "left":

        joints = LEFT_JOINT_IDS
        joint_names = LEFT_JOINT_NAMES

    elif arm == "right":

        joints = RIGHT_JOINT_IDS
        joint_names = RIGHT_JOINT_NAMES

    else:

        raise ValueError(
            f"Unknown arm: {arm}"
        )

    # --------------------------------------------------------
    # Save REAL starting state
    # --------------------------------------------------------

    start_qpos = data.qpos.copy()
    start_qvel = data.qvel.copy()

    # --------------------------------------------------------
    # Solve IK
    #
    # solve_ik changes data.qpos temporarily.
    # We only use its result to obtain target_q.
    # --------------------------------------------------------

    _, ik_error = solve_ik(
        data,
        target,
        arm
    )

    target_q = np.array([
        data.qpos[joint]
        for joint in joints
    ])

    # --------------------------------------------------------
    # Restore REAL starting state
    # --------------------------------------------------------

    data.qpos[:] = start_qpos
    data.qvel[:] = start_qvel

    mujoco.mj_forward(
        model,
        data
    )

    # --------------------------------------------------------
    # Send target joint positions
    # --------------------------------------------------------

    actuator_ids = [
        actuator_id(name)
        for name in joint_names
    ]

    for i, actuator in enumerate(actuator_ids):

        data.ctrl[actuator] = target_q[i]

    # Keep gripper open while moving
    if arm == "left":

        data.ctrl[
            LEFT_GRIPPER_ID
        ] = 0.5

    else:

        data.ctrl[
            RIGHT_GRIPPER_ID
        ] = 0.5

    # --------------------------------------------------------
    # Let actuators settle
    # --------------------------------------------------------

    for step in range(steps):

        mujoco.mj_step(
            model,
            data
        )

        # If an object is already grasped,
        # keep it attached to the gripper.
        if GRASPED_OBJECTS:

            update_grasped_objects(
                data
            )

    # --------------------------------------------------------
    # Final position
    # --------------------------------------------------------

    final_position = (
        get_gripper_position(
            data,
            arm
        )
    )

    final_error = np.linalg.norm(
        target - final_position
    )

    print(
        f"IK error: "
        f"{ik_error:.4f} m"
    )

    print(
        f"Final position:"
        f" x={final_position[0]:.3f}"
        f" y={final_position[1]:.3f}"
        f" z={final_position[2]:.3f}"
    )

    print(
        f"Final error: "
        f"{final_error:.4f} m"
    )

    return final_error

# ============================================================ DUAL ARM MOVEMENT
# ============================================================

def move_arm_pair(
    data,
    left_target,
    right_target
):

    left_error = move_arm(
        data,
        "left",
        np.asarray(left_target)
    )

    right_error = move_arm(
        data,
        "right",
        np.asarray(right_target)
    )

    return (
        left_error,
        right_error
    )


# ============================================================
# GRASP
# ============================================================

def grasp_object(
    data,
    object_name,
    arm
):

    object_position = (
        get_body_position(
            data,
            object_name
        )
    )

    gripper_position = (
        get_gripper_position(
            data,
            arm
        )
    )

    # Logical grasp:
    # Once grasped, the object follows the gripper directly.
    #
    # This simulation treats the gripperframe as the object's
    # logical attachment point.

    offset = np.zeros(3, dtype=float)

    GRASPED_OBJECTS[
        object_name
    ] = {
        "arm": arm,
        "offset": offset,
    }

    close_gripper(
        data,
        arm
    )

    print(
        f"GRASP: {object_name} with {arm} arm"
    )


# ============================================================
# RELEASE
# ============================================================

def release_object(
    data,
    object_name,
    arm
):

    # Make sure object is at its
    # latest gripper-relative position.
    update_grasped_objects(
        data
    )

    open_gripper(
        data,
        arm
    )

    if object_name in GRASPED_OBJECTS:

        del GRASPED_OBJECTS[
            object_name
        ]

    print(
        f"RELEASE: {object_name}"
        f" from {arm} arm"
    )


# ============================================================
# FREEJOINT POSITION
# ============================================================

def set_freejoint_position(
    data,
    joint_name,
    position
):

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name
    )

    if jid < 0:
        raise ValueError(
            f"Joint not found: {joint_name}"
        )

    qpos_address = (
        model.jnt_qposadr[jid]
    )

    data.qpos[
        qpos_address:
        qpos_address + 3
    ] = np.asarray(
        position,
        dtype=float
    )

    mujoco.mj_forward(
        model,
        data
    )


# ============================================================
# OBJECT VERIFICATION
# ============================================================

def verify_object(
    data,
    object_name,
    expected_position,
    tolerance=0.05
):

    actual = get_body_position(
        data,
        object_name
    )

    error = np.linalg.norm(
        np.asarray(expected_position)
        - actual
    )

    success = (
        error <= tolerance
    )

    print(
        f"VERIFY {object_name}:"
        f" error={error:.4f} m"
        f" | "
        f"{'PASS' if success else 'FAIL'}"
    )

    return success


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print(
        "======================================"
    )

    print(
        "SO-101 DUAL ARM CONTROLLER"
    )

    print(
        "======================================"
    )

    data = create_simulation()

    print(
        "\nLeft gripper:"
    )

    print(
        get_gripper_position(
            data,
            "left"
        )
    )

    print(
        "\nRight gripper:"
    )

    print(
        get_gripper_position(
            data,
            "right"
        )
    )

    print(
        "\nController loaded successfully."
    )
