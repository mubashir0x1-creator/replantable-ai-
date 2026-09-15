# RePlanTable AI
#
# AI assistance: ChatGPT was used to help draft the initial MuJoCo
# dual-arm control example.
#
# The project concept, task design, architecture, and implementation
# decisions are the student's own work.

from pathlib import Path

import mujoco
import numpy as np


MODEL_XML = r"""
<mujoco model="replantable_dual_arm">

    <option timestep="0.002" gravity="0 0 -9.81"/>

    <worldbody>

        <!-- Floor -->
        <geom
            name="floor"
            type="plane"
            size="5 5 0.1"
            pos="0 0 0"
        />


        <!-- Table -->
        <body name="table" pos="0 0 0.9">

            <geom
                name="table_top"
                type="box"
                size="1.4 0.8 0.05"
            />

            <geom
                name="leg_1"
                type="box"
                size="0.05 0.05 0.45"
                pos="-1.2 -0.65 -0.45"
            />

            <geom
                name="leg_2"
                type="box"
                size="0.05 0.05 0.45"
                pos="1.2 -0.65 -0.45"
            />

            <geom
                name="leg_3"
                type="box"
                size="0.05 0.05 0.45"
                pos="-1.2 0.65 -0.45"
            />

            <geom
                name="leg_4"
                type="box"
                size="0.05 0.05 0.45"
                pos="1.2 0.65 -0.45"
            />

        </body>


        <!-- ---------------- LEFT ARM ---------------- -->

        <body name="left_base" pos="-1.1 0 1.02">

            <joint
                name="left_joint_1"
                type="hinge"
                axis="0 1 0"
                pos="0 0 0"
                range="-1.5 1.5"
            />

            <geom
                type="capsule"
                fromto="0 0 0 -0.35 0 0"
                size="0.07"
            />


            <body name="left_link_2" pos="-0.35 0 0">

                <joint
                    name="left_joint_2"
                    type="hinge"
                    axis="0 1 0"
                    pos="0 0 0"
                    range="-2.2 2.2"
                />

                <geom
                    type="capsule"
                    fromto="0 0 0 -0.35 0 0"
                    size="0.06"
                />

                <site
                    name="left_gripper"
                    pos="-0.35 0 0"
                    size="0.05"
                />

            </body>

        </body>


        <!-- ---------------- RIGHT ARM ---------------- -->

        <body name="right_base" pos="1.1 0 1.02">

            <joint
                name="right_joint_1"
                type="hinge"
                axis="0 1 0"
                pos="0 0 0"
                range="-1.5 1.5"
            />

            <geom
                type="capsule"
                fromto="0 0 0 0.35 0 0"
                size="0.07"
            />


            <body name="right_link_2" pos="0.35 0 0">

                <joint
                    name="right_joint_2"
                    type="hinge"
                    axis="0 1 0"
                    pos="0 0 0"
                    range="-2.2 2.2"
                />

                <geom
                    type="capsule"
                    fromto="0 0 0 0.35 0 0"
                    size="0.06"
                />

                <site
                    name="right_gripper"
                    pos="0.35 0 0"
                    size="0.05"
                />

            </body>

        </body>


        <!-- Objects -->

        <body name="plate" pos="-0.25 0 1.08">

            <freejoint/>

            <geom
                type="cylinder"
                size="0.20 0.025"
                mass="0.2"
            />

        </body>


        <body name="cup" pos="0.25 0 1.20">

            <freejoint/>

            <geom
                type="cylinder"
                size="0.08 0.12"
                mass="0.15"
            />

        </body>


        <body name="fork" pos="0 0.35 1.07">

            <freejoint/>

            <geom
                type="box"
                size="0.03 0.12 0.01"
                mass="0.05"
            />

        </body>

    </worldbody>


    <!-- Position actuators -->

    <actuator>

        <position
            name="left_motor_1"
            joint="left_joint_1"
            kp="80"
            kv="10"
            ctrlrange="-1.5 1.5"
        />

        <position
            name="left_motor_2"
            joint="left_joint_2"
            kp="80"
            kv="10"
            ctrlrange="-2.2 2.2"
        />

        <position
            name="right_motor_1"
            joint="right_joint_1"
            kp="80"
            kv="10"
            ctrlrange="-1.5 1.5"
        />

        <position
            name="right_motor_2"
            joint="right_joint_2"
            kp="80"
            kv="10"
            ctrlrange="-2.2 2.2"
        />

    </actuator>

</mujoco>
"""


def body_position(model, data, name):
    """Return the world position of a body."""

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        name,
    )

    return data.xpos[body_id].copy()


def print_state(model, data, label):
    """Print the current robot and object state."""

    left = body_position(
        model,
        data,
        "left_link_2",
    )

    right = body_position(
        model,
        data,
        "right_link_2",
    )

    plate = body_position(
        model,
        data,
        "plate",
    )

    cup = body_position(
        model,
        data,
        "cup",
    )

    print(f"\n--- {label} ---")

    print(
        "Left arm:  "
        f"x={left[0]:.2f}, "
        f"y={left[1]:.2f}, "
        f"z={left[2]:.2f}"
    )

    print(
        "Right arm: "
        f"x={right[0]:.2f}, "
        f"y={right[1]:.2f}, "
        f"z={right[2]:.2f}"
    )

    print(
        "Plate:     "
        f"x={plate[0]:.2f}, "
        f"y={plate[1]:.2f}, "
        f"z={plate[2]:.2f}"
    )

    print(
        "Cup:       "
        f"x={cup[0]:.2f}, "
        f"y={cup[1]:.2f}, "
        f"z={cup[2]:.2f}"
    )


def main():
    """Create and control a simple dual-arm robot."""

    model = mujoco.MjModel.from_xml_string(
        MODEL_XML
    )

    data = mujoco.MjData(model)

    print("RePlanTable dual-arm simulation loaded.")

    print(
        f"Bodies: {model.nbody}"
    )

    print(
        f"Joints: {model.njnt}"
    )

    print(
        f"Actuators: {model.nu}"
    )

    print_state(
        model,
        data,
        "Initial state",
    )


    # Initial neutral position.

    data.ctrl[:] = 0.0

    for _ in range(300):

        mujoco.mj_step(
            model,
            data,
        )


    print_state(
        model,
        data,
        "Neutral pose",
    )


    # Move both arms inward.

    target = np.array(
        [
            -0.65,
            -0.35,
            0.65,
            -0.35,
        ],
        dtype=float,
    )

    print(
        "\nMoving both arms toward the workspace..."
    )

    for _ in range(1000):

        data.ctrl[:] = target

        mujoco.mj_step(
            model,
            data,
        )


    print_state(
        model,
        data,
        "After coordinated movement",
    )


    # Move both arms back.

    target = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    print(
        "\nReturning both arms to neutral..."
    )

    for _ in range(1000):

        data.ctrl[:] = target

        mujoco.mj_step(
            model,
            data,
        )


    print_state(
        model,
        data,
        "Final state",
    )


    print(
        "\nDual-arm control test completed."
    )


if __name__ == "__main__":
    main()
