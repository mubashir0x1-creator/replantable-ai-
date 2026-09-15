# RePlanTable AI
#
# Execution layer for:
# - Initial robot execution
# - Automatic recovery
# - State-aware replanning
#
# AI assistance: OpenAI ChatGPT was used to help design
# the verification-aware recovery execution flow.

import numpy as np

from simulation.so101.so101_dual_arm import (
    model,
    create_simulation as create_so101_simulation,
    move_arm,
    grasp_object,
    release_object,
    verify_object,
)

from simulation.coordinate_adapter import (
    planner_to_robot,
)


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

VERIFICATION_TOLERANCE = 0.05

# The plate has a known small movement offset in this
# simulation. This affects physical movement only.
#
# IMPORTANT:
# Semantic verification MUST still use planner_to_robot()
# without this compensation.
PLATE_MOVE_COMPENSATION_X = 0.0


# ------------------------------------------------------------
# SIMULATION
# ------------------------------------------------------------

def create_simulation():
    """
    Create the single MuJoCo simulation used for the
    entire voice session.

    run_planned_task.py expects:
        model, data = create_simulation()
    """

    data = create_so101_simulation()

    return model, data


# ------------------------------------------------------------
# TARGET CONVERSION
# ------------------------------------------------------------

def _is_planner_target(target):
    """
    Planner targets use z values around 1.08 / 1.20.

    Robot-space targets use z around 0.20.

    This lets us safely handle both:
      planner Action targets
      robot-space state-aware Action targets
    """

    target = np.asarray(
        target,
        dtype=float,
    )

    if target.shape != (3,):
        raise ValueError(
            f"Expected 3D target, got: {target}"
        )

    return float(target[2]) > 0.5


def _to_robot_target(target):
    """
    Convert a planner-space target into the SO-101
    simulation coordinate system.

    If the target is already in robot coordinates,
    leave it unchanged.
    """

    target = np.asarray(
        target,
        dtype=float,
    )

    if _is_planner_target(target):

        return planner_to_robot(
            target
        )

    return target.copy()


def _movement_target(
    object_name,
    target,
):
    """
    Return the physical movement target.

    The plate compensation is ONLY for robot movement.

    It must NOT be used as the semantic verification target.
    """

    robot_target = _to_robot_target(
        target
    )

    movement_target = robot_target.copy()

    if object_name == "plate":

        movement_target[0] += (
            PLATE_MOVE_COMPENSATION_X
        )

    return movement_target


def _verification_target(target):
    """
    Return the semantic verification target.

    NEVER apply movement compensation here.
    """

    return _to_robot_target(
        target
    )


# ------------------------------------------------------------
# ACTION HELPERS
# ------------------------------------------------------------

def _print_action(index, action):
    print(
        f"{index}. "
        f"{action.action} "
        f"{action.object_name} "
        f"WITH {action.arm.upper()} ARM "
        f"AT {action.target}"
    )


def _execute_pick(
    data,
    action,
):
    """
    Execute a PICK action.

    The action target may be planner-space or robot-space.
    """

    target = _to_robot_target(
        action.target
    )

    print()
    print(
        f"PICK {action.object_name} "
        f"WITH {action.arm.upper()} ARM"
    )

    print(
        "Robot pick target:",
        np.round(
            target,
            4,
        )
    )

    move_arm(
        data,
        action.arm,
        target,
    )

    grasp_object(
        data,
        action.object_name,
        action.arm,
    )


def _execute_place(
    data,
    action,
    force_failure=False,
):
    """
    Execute a PLACE action.

    Physical movement target and semantic verification
    target are intentionally kept separate.
    """

    verification_target = (
        _verification_target(
            action.target
        )
    )

    movement_target = (
        _movement_target(
            action.object_name,
            action.target,
        )
    )

    print()
    print(
        f"PLACE {action.object_name} "
        f"WITH {action.arm.upper()} ARM"
    )

    print(
        "Verification target:",
        np.round(
            verification_target,
            4,
        )
    )

    print(
        "Movement target:",
        np.round(
            movement_target,
            4,
        )
    )

    # --------------------------------------------------------
    # OPTIONAL FORCED FAILURE
    #
    # Used by the demo to prove that the automatic recovery
    # pipeline actually works.
    #
    # We intentionally move the cup to a wrong position.
    # Recovery will then use the real failed position.
    # --------------------------------------------------------

    actual_movement_target = (
        movement_target.copy()
    )

    if force_failure:

        print()
        print(
            "DEMO MODE: forcing placement failure."
        )

        actual_movement_target[0] += 0.10

        print(
            "Forced movement target:",
            np.round(
                actual_movement_target,
                4,
            )
        )

    move_arm(
        data,
        action.arm,
        actual_movement_target,
    )

    release_object(
        data,
        action.object_name,
        action.arm,
    )

    return verification_target


def _execute_plan_actions(
    data,
    plan,
    force_failure=False,
):
    """
    Execute PICK / PLACE actions in order.

    Returns:
        placement_targets
    """

    placement_targets = {}

    force_failure_used = False

    for index, action in enumerate(
        plan,
        start=1,
    ):

        action_name = str(
            action.action
        ).upper()

        print()
        print(
            "=" * 64
        )

        _print_action(
            index,
            action,
        )

        if action_name == "PICK":

            _execute_pick(
                data,
                action,
            )

        elif action_name == "PLACE":

            # Only force one failure. This keeps the demo
            # deterministic and prevents every object from
            # being deliberately broken.
            should_force_failure = (
                force_failure
                and not force_failure_used
            )

            _execute_place(
                data,
                action,
                force_failure=should_force_failure,
            )

            if should_force_failure:

                force_failure_used = True

            placement_targets[
                action.object_name
            ] = _verification_target(
                action.target
            )

        else:

            print(
                f"WARNING: Unknown action "
                f"'{action.action}'. Skipping."
            )

    return placement_targets


# ------------------------------------------------------------
# INITIAL PLAN
# ------------------------------------------------------------

def execute_initial_plan(
    model,
    data,
    plan,
    force_failure=False,
):
    """
    Execute the first planner-generated task.

    run_planned_task.py expects a dictionary containing:

        success
        world_state
        failed_objects
        failed_states
    """

    from state.world_state import WorldState

    world_state = WorldState()

    print()
    print("=" * 64)
    print("INITIAL PLAN EXECUTION")
    print("=" * 64)

    print()
    print(
        f"Actions: {len(plan)}"
    )

    # Execute actions.
    _execute_plan_actions(
        data=data,
        plan=plan,
        force_failure=force_failure,
    )

    # --------------------------------------------------------
    # Robot-level verification
    #
    # This verification is intentionally performed using
    # the semantic target, NOT the compensated movement target.
    # --------------------------------------------------------

    failed_objects = []
    failed_states = {}

    for action in plan:

        action_name = str(
            action.action
        ).upper()

        if action_name != "PLACE":
            continue

        object_name = action.object_name

        expected_position = (
            _verification_target(
                action.target
            )
        )

        result = verify_object(
            data,
            object_name,
            expected_position,
            tolerance=VERIFICATION_TOLERANCE,
        )

        if not result:

            if object_name not in failed_objects:

                failed_objects.append(
                    object_name
                )

                try:
                    actual_position = (
                        _get_object_position(
                            data,
                            object_name,
                        )
                    )
                except Exception:
                    actual_position = (
                        0.0,
                        0.0,
                        0.0,
                    )

                failed_states[
                    object_name
                ] = {
                    "position": actual_position,
                    "status": "failed",
                }

        else:

            try:
                actual_position = (
                    _get_object_position(
                        data,
                        object_name,
                    )
                )
            except Exception:
                actual_position = (
                    0.0,
                    0.0,
                    0.0,
                )

            world_state.mark_placed(
                object_name,
                actual_position,
            )

    # Mark failed objects in WorldState.
    for object_name in failed_objects:

        position = (
            failed_states[
                object_name
            ]["position"]
        )

        world_state.mark_failed(
            object_name,
            position,
        )

    success = (
        len(failed_objects) == 0
    )

    return {
        "success": success,
        "world_state": world_state,
        "failed_objects": failed_objects,
        "failed_states": failed_states,
    }


# ------------------------------------------------------------
# OBJECT POSITION
# ------------------------------------------------------------

def _get_object_position(
    data,
    object_name,
):
    """
    Get the current MuJoCo body position.

    Uses the SO-101 controller's public helper when possible.
    """

    from simulation.so101.so101_dual_arm import (
        get_body_position,
    )

    position = get_body_position(
        data,
        object_name,
    )

    return tuple(
        float(value)
        for value in position
    )


# ------------------------------------------------------------
# AUTOMATIC RECOVERY
# ------------------------------------------------------------

def execute_recovery(
    model,
    data,
    recovery_plan,
    world_state,
):
    """
    Execute an automatically generated recovery plan.

    run_planned_task.py expects:

        recovery_result["success"]
        recovery_result["world_state"]
    """

    print()
    print("=" * 64)
    print("RECOVERY EXECUTION")
    print("=" * 64)

    if not recovery_plan:

        print(
            "No recovery actions."
        )

        return {
            "success": True,
            "world_state": world_state,
        }

    try:

        _execute_plan_actions(
            data=data,
            plan=recovery_plan,
            force_failure=False,
        )

        # ----------------------------------------------------
        # Recovery-level robot verification.
        #
        # Vision verification happens separately in
        # run_planned_task.py.
        # ----------------------------------------------------

        failed_objects = []

        for action in recovery_plan:

            action_name = str(
                action.action
            ).upper()

            if action_name != "PLACE":
                continue

            object_name = action.object_name

            expected_position = (
                _verification_target(
                    action.target
                )
            )

            result = verify_object(
                data,
                object_name,
                expected_position,
                tolerance=VERIFICATION_TOLERANCE,
            )

            if result:

                actual_position = (
                    _get_object_position(
                        data,
                        object_name,
                    )
                )

                world_state.mark_placed(
                    object_name,
                    actual_position,
                )

                print(
                    f"[RECOVERY PASS] "
                    f"{object_name} recovered."
                )

            else:

                failed_objects.append(
                    object_name
                )

                actual_position = (
                    _get_object_position(
                        data,
                        object_name,
                    )
                )

                world_state.mark_failed(
                    object_name,
                    actual_position,
                )

                print(
                    f"[RECOVERY FAIL] "
                    f"{object_name} recovery failed."
                )

        success = (
            len(failed_objects) == 0
        )

        return {
            "success": success,
            "world_state": world_state,
            "failed_objects": failed_objects,
        }

    except Exception as exc:

        print()
        print(
            "RECOVERY EXECUTION ERROR:"
        )
        print(
            repr(exc)
        )

        return {
            "success": False,
            "world_state": world_state,
            "failed_objects": [],
            "error": str(exc),
        }


# ------------------------------------------------------------
# STATE-AWARE PLAN
# ------------------------------------------------------------

def execute_state_aware_plan(
    model,
    data,
    plan,
    world_state,
):
    """
    Execute a state-aware replanned action sequence.

    The Replanner may already have converted targets into
    robot-space coordinates, so target conversion is handled
    automatically by _to_robot_target().
    """

    print()
    print("=" * 64)
    print("STATE-AWARE PLAN EXECUTION")
    print("=" * 64)

    if not plan:

        return {
            "success": True,
            "world_state": world_state,
            "targets": {},
        }

    placement_targets = {}

    failed_objects = []

    try:

        # ----------------------------------------------------
        # Execute the state-aware plan.
        #
        # No forced failure here.
        # ----------------------------------------------------

        _execute_plan_actions(
            data=data,
            plan=plan,
            force_failure=False,
        )

        # ----------------------------------------------------
        # Robot verification.
        # ----------------------------------------------------

        for action in plan:

            action_name = str(
                action.action
            ).upper()

            if action_name != "PLACE":
                continue

            object_name = action.object_name

            verification_target = (
                _verification_target(
                    action.target
                )
            )

            placement_targets[
                object_name
            ] = tuple(
                float(value)
                for value in verification_target
            )

            result = verify_object(
                data,
                object_name,
                verification_target,
                tolerance=VERIFICATION_TOLERANCE,
            )

            if result:

                actual_position = (
                    _get_object_position(
                        data,
                        object_name,
                    )
                )

                world_state.mark_placed(
                    object_name,
                    actual_position,
                )

                print()
                print(
                    f"[STATE-AWARE PASS] "
                    f"{object_name}"
                )

            else:

                failed_objects.append(
                    object_name
                )

                actual_position = (
                    _get_object_position(
                        data,
                        object_name,
                    )
                )

                world_state.mark_failed(
                    object_name,
                    actual_position,
                )

                print()
                print(
                    f"[STATE-AWARE FAIL] "
                    f"{object_name}"
                )

        success = (
            len(failed_objects) == 0
        )

        return {
            "success": success,
            "world_state": world_state,
            "targets": placement_targets,
            "failed_objects": failed_objects,
        }

    except Exception as exc:

        print()
        print(
            "STATE-AWARE EXECUTION ERROR:"
        )
        print(
            repr(exc)
        )

        return {
            "success": False,
            "world_state": world_state,
            "targets": placement_targets,
            "failed_objects": failed_objects,
            "error": str(exc),
        }
