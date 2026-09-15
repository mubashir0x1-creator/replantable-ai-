# RePlanTable AI
#
# Natural Language
# -> Voice / Speechmatics
# -> Planner
# -> Robot
# -> Vision Observation
# -> Vision Verification
# -> Automatic Recovery
# -> State-Aware Replanning
# -> Continuous AI Voice Replanning
#
# AI assistance: OpenAI ChatGPT was used to help
# design the verification-aware execution flow.


from pathlib import Path
import sys


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from planner.task_planner import TaskPlanner

from planner.replanner import Replanner

from simulation.recovery_executor import (
    create_simulation,
    execute_initial_plan,
    execute_recovery,
    execute_state_aware_plan,
)

from simulation.coordinate_adapter import (
    planner_to_robot,
    robot_to_planner,
)

from vision.vision_pipeline import VisionPipeline

from voice.voice_input import get_voice_instruction


VOICE_AUDIO_FILES = [
    PROJECT_ROOT / "voice" / "ai_voice.mp3",
    PROJECT_ROOT / "voice" / "ai_voice_2.mp3",
    PROJECT_ROOT / "voice" / "ai_voice_3.mp3",
]


def print_result(title, result):

    print()
    print("=" * 64)
    print(title)
    print("=" * 64)

    if result["success"]:

        print(
            "PASS: Task verification successful."
        )

    else:

        print(
            "FAIL: Task verification failed."
        )

        failed_objects = result.get(
            "failed_objects",
            [],
        )

        if failed_objects:

            print()
            print("Failed objects:")

            for object_name in failed_objects:

                print(
                    f"- {object_name}"
                )


def get_targets_from_plan(plan):

    targets = {}

    for action in plan:

        action_name = str(
            action.action
        ).upper()

        if action_name != "PLACE":
            continue

        targets[action.object_name] = (
            action.target
        )

    return targets


def get_robot_targets_from_plan(plan):

    planner_targets = (
        get_targets_from_plan(plan)
    )

    robot_targets = {}

    for object_name, planner_target in (
        planner_targets.items()
    ):

        robot_target = planner_to_robot(
            planner_target
        )

        robot_targets[object_name] = (
            tuple(
                float(value)
                for value in robot_target
            )
        )

    return robot_targets


def run_vision_verification(
    vision,
    plan,
    world_state,
):

    planner_targets = (
        get_targets_from_plan(plan)
    )

    if not planner_targets:

        print()
        print(
            "VISION: No placement targets "
            "found in plan."
        )

        return True

    robot_targets = {}

    for object_name, planner_target in (
        planner_targets.items()
    ):

        robot_target = planner_to_robot(
            planner_target
        )

        robot_targets[object_name] = (
            tuple(
                float(value)
                for value in robot_target
            )
        )

        print()
        print(
            f"{object_name}:"
        )

        print(
            "  Planner target:",
            tuple(
                round(float(value), 3)
                for value in planner_target
            ),
        )

        print(
            "  Robot target:",
            tuple(
                round(float(value), 3)
                for value in robot_target
            ),
        )

    print()
    print("=" * 64)
    print("VISION POST-EXECUTION VERIFICATION")
    print("=" * 64)

    results = vision.verify(
        robot_targets
    )

    all_success = True

    for object_name, result in (
        results.items()
    ):

        status = (
            "PASS"
            if result["success"]
            else "FAIL"
        )

        print(
            f"{object_name}: {status} | "
            f"error={result['error']:.6f}"
        )

        actual_position = (
            result["actual_position"]
        )

        if result["success"]:

            world_state.mark_placed(
                object_name,
                actual_position,
            )

        else:

            all_success = False

            world_state.mark_failed(
                object_name,
                actual_position,
            )

    print("=" * 64)

    return all_success


def run_initial_recovery(
    model,
    data,
    replanner,
    result,
    world_state,
):

    if result["success"]:

        return world_state

    print()
    print("=" * 64)
    print("FAILURE DETECTED")
    print("=" * 64)

    print()
    print(
        "Automatic recovery will now begin."
    )

    failed_objects = result.get(
        "failed_objects",
        [],
    )

    failed_states = result.get(
        "failed_states",
        {},
    )

    if not failed_objects:

        print(
            "No failed objects were reported."
        )

        return world_state

    recovery_plan = (
        replanner.recovery_plan(
            failed_objects=failed_objects,
            failed_states=failed_states,
        )
    )

    print()
    print("=" * 64)
    print("RECOVERY PLAN")
    print("=" * 64)

    for index, action in enumerate(
        recovery_plan,
        start=1,
    ):

        print(
            f"{index}. "
            f"{action.action} "
            f"{action.object_name} "
            f"WITH {action.arm.upper()} ARM "
            f"AT {action.target}"
        )

    if not recovery_plan:

        print(
            "No recovery actions generated."
        )

        return world_state

    print()
    print(
        "EXECUTING RECOVERY PLAN..."
    )

    recovery_result = execute_recovery(
        model=model,
        data=data,
        recovery_plan=recovery_plan,
        world_state=world_state,
    )

    print_result(
        "RECOVERY RESULT",
        recovery_result,
    )

    return recovery_result[
        "world_state"
    ]


def verify_recovery(
    vision,
    recovery_plan,
    world_state,
):

    if not recovery_plan:

        return True

    print()
    print(
        "RUNNING VISION VERIFICATION "
        "AFTER RECOVERY..."
    )

    recovery_success = (
        run_vision_verification(
            vision=vision,
            plan=recovery_plan,
            world_state=world_state,
        )
    )

    print()
    print("=" * 64)
    print("WORLD STATE AFTER RECOVERY VISION")
    print("=" * 64)

    world_state.show_state()

    return recovery_success


def process_next_voice_instruction(
    voice_number,
    audio_file,
    planner,
    replanner,
    model,
    data,
    vision,
    world_state,
):

    print()
    print("=" * 64)
    print(
        f"AI VOICE INPUT {voice_number}"
    )
    print("=" * 64)

    print()
    print(
        f"Audio file: {audio_file.name}"
    )

    print()
    print(
        "Transcribing AI voice instruction..."
    )

    instruction = get_voice_instruction(
        audio_file
    )

    if not instruction:

        print(
            "No instruction provided."
        )

        return world_state

    print()
    print(
        "VOICE INSTRUCTION:"
    )

    print(
        instruction
    )

    new_plan = planner.parse_instruction(
        instruction
    )

    if not new_plan:

        print()
        print(
            "ERROR: Could not understand "
            "the voice instruction."
        )

        return world_state

    print()
    print(
        f"PLAN FOR AI VOICE INPUT "
        f"{voice_number}"
    )

    planner.show_plan()

    print()
    print("=" * 64)
    print("STATE-AWARE REPLANNING")
    print("=" * 64)

    state_aware_plan = (
        replanner.state_aware_replan(
            new_plan,
            world_state,
        )
    )

    if not state_aware_plan:

        print()
        print(
            "NO MOVEMENT REQUIRED"
        )

        print()
        print(
            "The current world state already "
            "satisfies the new instruction."
        )

        world_state.show_state()

        return world_state

    print()
    print(
        "STATE-AWARE EXECUTION PLAN"
    )

    for index, action in enumerate(
        state_aware_plan,
        start=1,
    ):

        print(
            f"{index}. "
            f"{action.action} "
            f"{action.object_name} "
            f"WITH {action.arm.upper()} ARM "
            f"AT {action.target}"
        )

    print()
    print(
        "EXECUTING STATE-AWARE PLAN..."
    )

    execution_result = (
        execute_state_aware_plan(
            model=model,
            data=data,
            plan=state_aware_plan,
            world_state=world_state,
        )
    )

    print_result(
        f"VOICE {voice_number} EXECUTION RESULT",
        execution_result,
    )

    world_state = (
        execution_result[
            "world_state"
        ]
    )

    print()
    print(
        "RUNNING VISION VERIFICATION..."
    )

    vision_success = (
        run_vision_verification(
            vision=vision,
            plan=state_aware_plan,
            world_state=world_state,
        )
    )

    print()
    print("=" * 64)
    print(
        f"WORLD STATE AFTER VOICE "
        f"{voice_number}"
    )
    print("=" * 64)

    world_state.show_state()

    if vision_success:

        print()
        print(
            f"VOICE {voice_number}: "
            "VISION VERIFICATION PASS"
        )

        print()
        print(
            f"VOICE {voice_number}: "
            "FINAL STATUS PASS"
        )

    else:

        print()
        print(
            f"VOICE {voice_number}: "
            "VISION VERIFICATION FAILED"
        )

        print(
            "The failed object remains marked "
            "in WorldState for future replanning."
        )

        print()
        print(
            f"VOICE {voice_number}: "
            "FINAL STATUS FAIL"
        )

    return world_state


def main():

    print()
    print("=" * 64)
    print("RePlanTable AI")
    print(
        "AI VOICE + VISION-AWARE "
        "CONTINUOUS REPLANNING"
    )
    print("=" * 64)

    planner = TaskPlanner()

    replanner = Replanner(
        planner
    )

    print()
    print("=" * 64)
    print("AI VOICE SESSION")
    print("=" * 64)

    print()
    print(
        "Voice files to process:"
    )

    for audio_file in VOICE_AUDIO_FILES:

        print(
            f"- {audio_file.name}"
        )

        if not audio_file.exists():

            print(
                f"ERROR: File not found: "
                f"{audio_file}"
            )

            return

    print()
    print("=" * 64)
    print("AI VOICE INPUT 1")
    print("=" * 64)

    print()
    print(
        "Transcribing AI voice instruction..."
    )

    instruction = get_voice_instruction(
        VOICE_AUDIO_FILES[0]
    )

    if not instruction:

        print(
            "No instruction provided."
        )

        return

    print()
    print(
        "VOICE INSTRUCTION:"
    )

    print(
        instruction
    )

    plan = planner.parse_instruction(
        instruction
    )

    if not plan:

        print()
        print(
            "ERROR: I could not understand "
            "the voice instruction."
        )

        print()
        print(
            "The transcript was:"
        )

        print(
            instruction
        )

        return

    print()
    print("=" * 64)
    print("TASK PLAN")
    print("=" * 64)

    planner.show_plan()

    print()
    print(
        "CREATING ONE SIMULATION FOR "
        "THE ENTIRE SESSION..."
    )

    model, data = create_simulation()

    vision = VisionPipeline(
        model,
        data,
    )

    print()
    print(
        "RUNNING INITIAL VISION OBSERVATION..."
    )

    vision.print_observation()

    print()
    print(
        "EXECUTING FIRST PLAN..."
    )

    result = execute_initial_plan(
        model=model,
        data=data,
        plan=plan,
        force_failure=True,
    )

    print_result(
        "FIRST PLAN RESULT",
        result,
    )

    world_state = result[
        "world_state"
    ]

    print()
    print(
        "RUNNING VISION VERIFICATION "
        "AFTER FIRST EXECUTION..."
    )

    vision_success = (
        run_vision_verification(
            vision=vision,
            plan=plan,
            world_state=world_state,
        )
    )

    print()
    print("=" * 64)
    print(
        "WORLD STATE AFTER VISION"
    )
    print("=" * 64)

    world_state.show_state()

    recovery_required = (
        not result["success"]
        or not vision_success
    )

    if recovery_required:

        print()
        print("=" * 64)
        print("FAILURE DETECTED")
        print("=" * 64)

        print()
        print(
            "Automatic recovery will now begin."
        )

        vision_failed_objects = []

        for object_name in (
            world_state.objects
        ):

            if (
                world_state.get_status(
                    object_name
                )
                == "failed"
            ):

                vision_failed_objects.append(
                    object_name
                )

        if vision_failed_objects:

            result["success"] = False

            result["failed_objects"] = (
                vision_failed_objects
            )

            result["failed_states"] = {
                object_name: world_state.get_object(
                    object_name
                )
                for object_name in (
                    vision_failed_objects
                )
            }

        failed_objects = result.get(
            "failed_objects",
            [],
        )

        failed_states = result.get(
            "failed_states",
            {},
        )

        print()
        print(
            "=" * 64
        )

        print(
            "AUTOMATIC RECOVERY"
        )

        print(
            "=" * 64
        )

        if failed_objects:

            recovery_plan = (
                replanner.recovery_plan(
                    failed_objects=failed_objects,
                    failed_states=failed_states,
                )
            )

            print()
            print(
                "RECOVERY PLAN"
            )

            print(
                "=" * 64
            )

            for index, action in enumerate(
                recovery_plan,
                start=1,
            ):

                print(
                    f"{index}. "
                    f"{action.action} "
                    f"{action.object_name} "
                    f"WITH {action.arm.upper()} ARM "
                    f"AT {action.target}"
                )

            if recovery_plan:

                print()
                print(
                    "EXECUTING RECOVERY PLAN..."
                )

                recovery_result = execute_recovery(
                    model=model,
                    data=data,
                    recovery_plan=recovery_plan,
                    world_state=world_state,
                )

                if isinstance(
                    recovery_result,
                    dict,
                ):

                    robot_recovery_success = bool(
                        recovery_result.get(
                            "success",
                            False,
                        )
                    )

                    if recovery_result.get(
                        "world_state"
                    ) is not None:

                        world_state = (
                            recovery_result[
                                "world_state"
                            ]
                        )

                else:

                    robot_recovery_success = bool(
                        recovery_result
                    )


                if robot_recovery_success:

                    print()
                    print(
                        "ROBOT RECOVERY: PASS"
                    )

                else:

                    print()
                    print(
                        "ROBOT RECOVERY: FAIL"
                    )


                print()
                print(
                    "RUNNING VISION VERIFICATION "
                    "AFTER RECOVERY..."
                )

                recovery_vision_success = (
                    run_vision_verification(
                        vision=vision,
                        plan=recovery_plan,
                        world_state=world_state,
                    )
                )


                print()
                print("=" * 64)
                print(
                    "WORLD STATE AFTER RECOVERY"
                )
                print("=" * 64)

                world_state.show_state()


                if recovery_vision_success:

                    print()
                    print(
                        "VISION RECOVERY: PASS"
                    )

                else:

                    print()
                    print(
                        "VISION RECOVERY: FAIL"
                    )


                final_recovery_success = (
                    robot_recovery_success
                    and recovery_vision_success
                )


                result["success"] = (
                    final_recovery_success
                )


                print()
                print("=" * 64)
                print(
                    "FINAL RECOVERY STATUS"
                )
                print("=" * 64)


                if final_recovery_success:

                    print(
                        "ROBOT VERIFICATION: PASS"
                    )

                    print(
                        "VISION VERIFICATION: PASS"
                    )

                    print(
                        "FINAL TASK STATUS: PASS"
                    )

                else:

                    print(
                        "ROBOT VERIFICATION: "
                        + (
                            "PASS"
                            if robot_recovery_success
                            else "FAIL"
                        )
                    )

                    print(
                        "VISION VERIFICATION: "
                        + (
                            "PASS"
                            if recovery_vision_success
                            else "FAIL"
                        )
                    )

                    print(
                        "FINAL TASK STATUS: FAIL"
                    )

            else:

                result["success"] = False

                print()
                print(
                    "RECOVERY PLAN: EMPTY"
                )

                print()
                print(
                    "FINAL TASK STATUS: FAIL"
                )

        else:

            result["success"] = False

            print()
            print(
                "No failed objects available "
                "for recovery."
            )

            print()
            print(
                "FINAL TASK STATUS: FAIL"
            )

    else:

        result["success"] = True

        print()
        print(
            "ROBOT VERIFICATION: PASS"
        )

        print()
        print(
            "VISION VERIFICATION: PASS"
        )

        print()
        print(
            "FIRST PLAN COMPLETED SUCCESSFULLY."
        )

        print()
        print(
            "FINAL TASK STATUS: PASS"
        )


    print()
    print("=" * 64)
    print("AI VOICE INPUT 2")
    print("=" * 64)


    world_state = (
        process_next_voice_instruction(
            voice_number=2,
            audio_file=VOICE_AUDIO_FILES[1],
            planner=planner,
            replanner=replanner,
            model=model,
            data=data,
            vision=vision,
            world_state=world_state,
        )
    )


    print()
    print("=" * 64)
    print("AI VOICE INPUT 3")
    print("=" * 64)


    world_state = (
        process_next_voice_instruction(
            voice_number=3,
            audio_file=VOICE_AUDIO_FILES[2],
            planner=planner,
            replanner=replanner,
            model=model,
            data=data,
            vision=vision,
            world_state=world_state,
        )
    )


    print()
    print("=" * 64)
    print("FINAL WORLD STATE")
    print("=" * 64)

    world_state.show_state()


    print()
    print("=" * 64)
    print("SESSION COMPLETE")
    print("=" * 64)


if __name__ == "__main__":
    main()
