import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from vision.live_detector import LiveSceneDetector


class VisionVerifier:
    def __init__(self, model, data, tolerance=0.01):
        self.detector = LiveSceneDetector(model, data)
        self.tolerance = tolerance

    def verify_object(self, object_name, target_position):
        detection = self.detector.detect_object(object_name)

        actual_position = np.asarray(
            detection["position"],
            dtype=float
        )

        target_position = np.asarray(
            target_position,
            dtype=float
        )

        error = float(
            np.linalg.norm(actual_position - target_position)
        )

        success = error <= self.tolerance

        return {
            "object": object_name,
            "detected": detection["detected"],
            "actual_position": tuple(actual_position),
            "target_position": tuple(target_position),
            "error": error,
            "success": success,
        }

    def verify_scene(self, targets):
        results = {}

        for object_name, target_position in targets.items():
            results[object_name] = self.verify_object(
                object_name,
                target_position
            )

        return results


if __name__ == "__main__":
    from simulation.recovery_executor import create_simulation

    model, data = create_simulation()

    verifier = VisionVerifier(model, data)

    targets = {
        "plate": (-0.45, 0.0, 1.08),
        "cup": (0.45, 0.0, 1.20),
    }

    results = verifier.verify_scene(targets)

    print("=" * 64)
    print("VISION VERIFICATION")
    print("=" * 64)

    for object_name, result in results.items():
        status = "PASS" if result["success"] else "FAIL"

        print(
            f"{object_name}: {status} | "
            f"error={result['error']:.6f}"
        )

    print("=" * 64)
