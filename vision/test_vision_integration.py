import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import mujoco

from simulation.recovery_executor import create_simulation
from simulation.bimanual_manipulation import (
    set_freejoint_position,
)
from vision.live_detector import LiveSceneDetector
from vision.verifier import VisionVerifier
from state.world_state import WorldState


def main():
    model, data = create_simulation()

    detector = LiveSceneDetector(model, data)
    verifier = VisionVerifier(model, data)
    world_state = WorldState()

    print("=" * 64)
    print("VISION INTEGRATION TEST")
    print("=" * 64)

    # ---------------------------------------------------------
    # STEP 1: Detect initial scene
    # ---------------------------------------------------------
    print("\n[1] INITIAL VISION DETECTION")

    scene = detector.detect_scene()

    for object_name, info in scene.items():
        print(
            f"{object_name}: "
            f"x={info['position'][0]:.3f} "
            f"y={info['position'][1]:.3f} "
            f"z={info['position'][2]:.3f}"
        )

    # ---------------------------------------------------------
    # STEP 2: Update WorldState
    # ---------------------------------------------------------
    print("\n[2] UPDATE WORLD STATE")

    detector.update_world_state(world_state)
    world_state.show_state()

    # ---------------------------------------------------------
    # STEP 3: Move cup to its RIGHT target
    # ---------------------------------------------------------
    print("\n[3] SIMULATING CUP PLACEMENT")

    target = (0.44, 0.0, 1.20)

    set_freejoint_position(
        model,
        data,
         "cup_freejoint",
        target
    )

    mujoco.mj_forward(model, data)

    # ---------------------------------------------------------
    # STEP 4: Vision verification
    # ---------------------------------------------------------
    print("\n[4] VISION VERIFICATION")

    result = verifier.verify_object(
         "cup",
        target
    )

    status = "PASS" if result["success"] else "FAIL"

    print(
        f"cup: {status} | "
        f"error={result['error']:.6f}"
    )

    # ---------------------------------------------------------
    # STEP 5: Update WorldState after verification
    # ---------------------------------------------------------
    print("\n[5] WORLD STATE AFTER VERIFICATION")

    if result["success"]:
        world_state.mark_placed(
             "cup",
            result["actual_position"]
        )
    else:
        world_state.mark_failed(
             "cup_freejoint",
            result["actual_position"]
        )

    world_state.show_state()

    print("=" * 64)

    if result["success"]:
        print("VISION INTEGRATION: PASS")
    else:
        print("VISION INTEGRATION: FAIL")

    print("=" * 64)


if __name__ == "__main__":
    main()
