from pathlib import Path
import sys


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from state.world_state import WorldState


def main():

    state = WorldState()

    print()
    print("INITIAL STATE")

    state.show_state()

    print()
    print("UPDATING PLATE")

    state.mark_placed(
        "plate",
        (-0.42, 0.0, 1.08),
    )

    print()
    print("UPDATING CUP")

    state.mark_placed(
        "cup",
        (0.44, 0.0, 1.20),
    )

    state.show_state()

    print()
    print("CHECKING CUP")

    print(
        "Cup position:",
        state.get_position("cup"),
    )

    print(
        "Cup status:",
        state.get_status("cup"),
    )

    print(
        "Cup distance from target:",
        state.distance_from(
            "cup",
            (0.44, 0.0, 1.20),
        ),
    )


if __name__ == "__main__":
    main()
