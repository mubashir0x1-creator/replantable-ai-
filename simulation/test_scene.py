# RePlanTable AI
#
# AI assistance: ChatGPT was used to help draft the initial MuJoCo
# test scene and Python structure.
#
# The project concept, experiment design, architecture, and implementation
# decisions are the student's own work.

from pathlib import Path

import mujoco
import numpy as np


XML = r"""
<mujoco model="replantable">

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
                pos="0 0 0"
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


        <!-- Red plate -->
        <body name="red_plate" pos="-0.45 0 1.02">

            <freejoint/>

            <geom
                name="red_plate_geom"
                type="cylinder"
                size="0.22 0.025"
            />

        </body>


        <!-- Blue cup -->
        <body name="blue_cup" pos="0.45 0 1.12">

            <freejoint/>

            <geom
                name="blue_cup_geom"
                type="cylinder"
                size="0.08 0.12"
            />

        </body>


        <!-- Green cube -->
        <body name="green_cube" pos="0 0.35 1.08">

            <freejoint/>

            <geom
                name="green_cube_geom"
                type="box"
                size="0.08 0.08 0.08"
            />

        </body>


        <!-- Left arm base -->
        <body name="left_arm_base" pos="-1.15 0 1.05">

            <geom
                name="left_base"
                type="cylinder"
                size="0.12 0.1"
            />

        </body>


        <!-- Right arm base -->
        <body name="right_arm_base" pos="1.15 0 1.05">

            <geom
                name="right_base"
                type="cylinder"
                size="0.12 0.1"
            />

        </body>

    </worldbody>

</mujoco>
"""


def main():
    """Load the first RePlanTable simulation model."""

    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    print("RePlanTable simulation loaded.")
    print(f"Number of bodies: {model.nbody}")
    print(f"Number of joints: {model.njnt}")

    print("\nObjects:")

    for body_name in [
        "red_plate",
        "blue_cup",
        "green_cube",
        "left_arm_base",
        "right_arm_base",
    ]:
        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )

        position = data.xpos[body_id]

        print(
            f"- {body_name}: "
            f"({position[0]:.2f}, "
            f"{position[1]:.2f}, "
            f"{position[2]:.2f})"
        )

    print("\nRunning 100 simulation steps...")

    for _ in range(100):
        mujoco.mj_step(model, data)

    print("Simulation completed successfully.")


if __name__ == "__main__":
    main()
