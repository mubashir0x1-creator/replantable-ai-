# RePlanTable AI
#
# World state tracking.
#
# WorldState stores actual MuJoCo/SO-101 world coordinates.
#
# AI assistance: OpenAI ChatGPT was used to help design
# the initial world-state tracking structure.


import numpy as np


class WorldState:

    def __init__(self):

        self.objects = {
            "plate": {
                "position": (
                    -0.05,
                    0.0,
                    0.20,
                ),
                "status": "initial",
            },

            "cup": {
                "position": (
                    0.05,
                    0.0,
                    0.20,
                ),
                "status": "initial",
            },
        }

    def update_object(
        self,
        object_name,
        position,
        status,
    ):

        if object_name not in self.objects:
            raise ValueError(
                f"Unknown object: {object_name}"
            )

        position = tuple(
            float(value)
            for value in position
        )

        self.objects[object_name] = {
            "position": position,
            "status": status,
        }

    def get_position(
        self,
        object_name,
    ):

        if object_name not in self.objects:
            raise ValueError(
                f"Unknown object: {object_name}"
            )

        return self.objects[
            object_name
        ]["position"]

    def get_status(
        self,
        object_name,
    ):

        if object_name not in self.objects:
            raise ValueError(
                f"Unknown object: {object_name}"
            )

        return self.objects[
            object_name
        ]["status"]

    def get_object(
        self,
        object_name,
    ):

        if object_name not in self.objects:
            raise ValueError(
                f"Unknown object: {object_name}"
            )

        return dict(
            self.objects[object_name]
        )

    def mark_placed(
        self,
        object_name,
        position,
    ):

        self.update_object(
            object_name=object_name,
            position=position,
            status="placed",
        )

    def mark_failed(
        self,
        object_name,
        position,
    ):

        self.update_object(
            object_name=object_name,
            position=position,
            status="failed",
        )

    def show_state(self):

        print()
        print("=" * 64)
        print("CURRENT WORLD STATE")
        print("=" * 64)

        for (
            object_name,
            state,
        ) in self.objects.items():

            position = state[
                "position"
            ]

            status = state[
                "status"
            ]

            print(
                f"{object_name}: "
                f"x={position[0]:.3f} "
                f"y={position[1]:.3f} "
                f"z={position[2]:.3f} "
                f"| status={status}"
            )

        print("=" * 64)

    def distance_from(
        self,
        object_name,
        target,
    ):

        position = np.asarray(
            self.get_position(
                object_name
            ),
            dtype=float,
        )

        target = np.asarray(
            target,
            dtype=float,
        )

        return float(
            np.linalg.norm(
                position - target
            )
        )
