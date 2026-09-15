import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import mujoco

from simulation.bimanual_manipulation import MODEL_XML
from state.world_state import WorldState


class SceneDetector:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(MODEL_XML)
        self.data = mujoco.MjData(self.model)

    def get_object_position(self, object_name):
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            object_name
        )

        if body_id == -1:
            raise ValueError(f"Unknown object: {object_name}")

        return tuple(
            float(value)
            for value in self.data.xpos[body_id]
        )

    def detect_scene(self):
        mujoco.mj_forward(self.model, self.data)

        detections = {}

        for object_name in ["plate", "cup"]:
            position = self.get_object_position(object_name)

            detections[object_name] = {
                "detected": True,
                "position": position,
            }

        return detections

    def create_world_state(self):
        scene = self.detect_scene()

        world_state = WorldState()

        for object_name, info in scene.items():
            if info["detected"]:
                world_state.update_object(
                    object_name,
                    info["position"],
                    "detected"
                )

        return world_state


if __name__ == "__main__":
    detector = SceneDetector()

    world_state = detector.create_world_state()

    print("=" * 64)
    print("VISION → WORLD STATE")
    print("=" * 64)

    world_state.show_state()

    print("=" * 64)
