from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from planner.task_planner import TaskPlanner
from planner.replanner import Replanner
from state.world_state import WorldState


def show_plan(title, plan):

    print()
    print("=" * 64)
    print(title)
    print("=" * 64)

    if not plan:
        print("NO ACTIONS REQUIRED.")
        return

    for number, action in enumerate(plan, start=1):

        print(
            f"{number}. "
            f"{action.action.upper()} "
            f"{action.object_name} "
            f"WITH {action.arm.upper()} ARM "
            f"AT {action.target}"
        )


def main():

    planner = TaskPlanner()
    replanner = Replanner(planner)
    world_state = WorldState()

    # Simulate the state after our successful recovery.
    world_state.mark_placed(
        "plate",
        (-0.42, 0.0, 1.08),
    )

    world_state.mark_placed(
        "cup",
        (0.44, 0.0, 1.20),
    )

    world_state.show_state()

    # --------------------------------------------------------
    # TEST 1
    # Plate is already at center.
    # Cup must move to left.
    # --------------------------------------------------------

    print()
    print("=" * 64)
    print("TEST 1")
    print("=" * 64)

    instruction = (
        "Keep the plate in the center "
        "and move the cup to the left."
    )

    new_plan = planner.parse_instruction(
        instruction
    )

    state_aware_plan = (
        replanner.state_aware_replan(
            new_plan,
            world_state,
        )
    )

    show_plan(
        "STATE-AWARE PLAN — TEST 1",
        state_aware_plan,
    )

    # --------------------------------------------------------
    # TEST 2
    # Both objects are already at requested positions.
    # --------------------------------------------------------

    print()
    print("=" * 64)
    print("TEST 2")
    print("=" * 64)

    instruction = (
        "Put the plate in the center "
        "and keep the cup on the right."
    )

    new_plan = planner.parse_instruction(
        instruction
    )

    state_aware_plan = (
        replanner.state_aware_replan(
            new_plan,
            world_state,
        )
    )

    show_plan(
        "STATE-AWARE PLAN — TEST 2",
        state_aware_plan,
    )

    # --------------------------------------------------------
    # TEST 3
    # Both objects should require movement.
    # --------------------------------------------------------

    print()
    print("=" * 64)
    print("TEST 3")
    print("=" * 64)

    instruction = (
        "Move the plate to the left "
        "and move the cup to the left."
    )

    new_plan = planner.parse_instruction(
        instruction
    )

    state_aware_plan = (
        replanner.state_aware_replan(
            new_plan,
            world_state,
        )
    )

    show_plan(
        "STATE-AWARE PLAN — TEST 3",
        state_aware_plan,
    )


if __name__ == "__main__":
    main()
