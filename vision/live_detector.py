import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import mujoco

from simulation.bimanual_manipulation import (
    MODEL_XML,
    get_body_position,
)
from state.world_state import WorldState


class LiveSceneDetector:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def detect_object(self, object_name):
        position = get_body_position(
            self.model,
            self.data,
            object_name
        )

        return {
            "object": object_name,
            "detected": True,
            "position": tuple(float(x) for x in position),
        }

    def detect_scene(self):
        mujoco.mj_forward(self.model, self.data)

        detections = {}

        for object_name in ["plate", "cup"]:
            detections[object_name] = self.detect_object(
                object_name
            )

        return detections

    def update_world_state(self, world_state):
        detections = self.detect_scene()

        for object_name, info in detections.items():
            if info["detected"]:
                current_status = world_state.get_status(
                    object_name
                )

                world_state.update_object(
                    object_name,
                    info["position"],
                    current_status
                )

        return world_state


if __name__ == "__main__":
    from simulation.recovery_executor import create_simulation

    model, data = create_simulation()

    detector = LiveSceneDetector(model, data)

    world_state = WorldState()

    detector.update_world_state(world_state)

    print("=" * 64)
    print("LIVE VISION → WORLD STATE")
    print("=" * 64)

    world_state.show_state()

    print("=" * 64)
