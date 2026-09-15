import sys
import os


# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)


if PROJECT_ROOT not in sys.path:

    sys.path.insert(
        0,
        PROJECT_ROOT,
    )


from vision.live_detector import LiveSceneDetector
from vision.verifier import VisionVerifier


# ------------------------------------------------------------
# VISION PIPELINE
# ------------------------------------------------------------

class VisionPipeline:
    """
    Connects the live MuJoCo scene with vision detection
    and verification.

    This module observes and verifies the current scene.
    It does not control the robot.
    """

    # The robot verification currently treats a 5 cm
    # positional error as acceptable. Vision uses the same
    # threshold so both verification layers remain consistent.

    DEFAULT_TOLERANCE = 0.05


    def __init__(
        self,
        model,
        data,
        tolerance=DEFAULT_TOLERANCE,
    ):

        self.tolerance = tolerance

        self.detector = LiveSceneDetector(
            model,
            data,
        )

        self.verifier = VisionVerifier(
            model,
            data,
            tolerance=self.tolerance,
        )


    # --------------------------------------------------------
    # OBSERVATION
    # --------------------------------------------------------

    def observe(self):
        """
        Read the current simulation scene.
        """

        return self.detector.detect_scene()


    # --------------------------------------------------------
    # VERIFICATION
    # --------------------------------------------------------

    def verify(self, targets):
        """
        Verify objects against their target positions.
        """

        return self.verifier.verify_scene(
            targets
        )


    # --------------------------------------------------------
    # PRINT OBSERVATION
    # --------------------------------------------------------

    def print_observation(self):
        """
        Print the current scene in a human-readable format.
        """

        scene = self.observe()

        print()
        print("=" * 64)
        print("VISION OBSERVATION")
        print("=" * 64)

        for object_name, info in scene.items():

            position = info["position"]

            print(
                f"{object_name}: "
                f"detected={info['detected']} | "
                f"x={position[0]:.3f} "
                f"y={position[1]:.3f} "
                f"z={position[2]:.3f}"
            )

        print("=" * 64)


    # --------------------------------------------------------
    # PRINT VERIFICATION
    # --------------------------------------------------------

    def print_verification(self, targets):
        """
        Print verification results.
        """

        results = self.verify(
            targets
        )

        print()
        print("=" * 64)
        print("VISION VERIFICATION RESULTS")
        print("=" * 64)

        print(
            f"Verification tolerance: "
            f"{self.tolerance:.3f} m"
        )

        for object_name, result in results.items():

            status = (
                "PASS"
                if result["success"]
                else "FAIL"
            )

            print(
                f"{object_name}: {status} | "
                f"error={result['error']:.6f}"
            )

        print("=" * 64)

        return results


# ------------------------------------------------------------
# DIRECT TEST
# ------------------------------------------------------------

if __name__ == "__main__":

    from simulation.recovery_executor import (
        create_simulation,
    )


    model, data = create_simulation()


    vision = VisionPipeline(
        model,
        data,
    )


    vision.print_observation()


    targets = {
        "plate": (
            -0.45,
            0.0,
            1.08,
        ),

        "cup": (
            0.45,
            0.0,
            1.20,
        ),
    }


    vision.print_verification(
        targets
    )
