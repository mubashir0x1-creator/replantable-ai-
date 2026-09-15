# RePlanTable AI
#
# Replanning and recovery planning.
#
# AI assistance: OpenAI ChatGPT was used to help design
# the recovery and state-aware replanning logic.

import numpy as np

from planner.task_planner import Action
from simulation.coordinate_adapter import planner_to_robot


class Replanner:

    def __init__(self, planner):

        self.planner = planner


    # --------------------------------------------------------
    # NORMAL REPLAN
    # --------------------------------------------------------

    def replan(self, new_instruction):

        return self.planner.parse_instruction(
            new_instruction
        )


    # --------------------------------------------------------
    # RECOVERY PLAN
    # --------------------------------------------------------

    def recovery_plan(
        self,
        failed_objects,
        failed_states,
    ):

        recovery_actions = []

        for object_name in failed_objects:

            if object_name not in failed_states:
                continue

            state = failed_states[
                object_name
            ]

            failed_position = tuple(
                float(value)
                for value in state["position"]
            )

            place_target = None

            for action in reversed(
                self.planner.current_plan
            ):

                if (
                    action.object_name
                    == object_name
                    and
                    str(action.action).upper()
                    == "PLACE"
                ):

                    place_target = action.target
                    break

            if place_target is None:
                continue

            arm = (
                "left"
                if object_name == "plate"
                else "right"
            )

            recovery_actions.append(
                Action(
                    action="PICK",
                    object_name=object_name,
                    arm=arm,
                    target=failed_position,
                )
            )

            recovery_actions.append(
                Action(
                    action="PLACE",
                    object_name=object_name,
                    arm=arm,
                    target=place_target,
                )
            )

        return recovery_actions


    # --------------------------------------------------------
    # STATE-AWARE REPLAN
    # --------------------------------------------------------

    def state_aware_replan(
        self,
        new_plan,
        world_state,
    ):

        result = []

        grouped = {}

        for action in new_plan:

            grouped.setdefault(
                action.object_name,
                []
            ).append(action)


        for object_name, actions in grouped.items():

            place_action = None
            pick_action = None

            for action in actions:

                action_name = str(
                    action.action
                ).upper()

                if action_name == "PICK":

                    pick_action = action

                elif action_name == "PLACE":

                    place_action = action


            if place_action is None:
                continue


            # ------------------------------------------------
            # CURRENT REAL WORLD POSITION
            # ------------------------------------------------

            current_position = np.asarray(
                world_state.get_position(
                    object_name
                ),
                dtype=float,
            )


            # ------------------------------------------------
            # PLANNER TARGET
            # ------------------------------------------------

            planner_target = np.asarray(
                place_action.target,
                dtype=float,
            )


            # ------------------------------------------------
            # CONVERT PLANNER TARGET TO ROBOT SPACE
            # ------------------------------------------------

            robot_target = planner_to_robot(
                planner_target
            )


            print()
            print(
                f"STATE CHECK: {object_name}"
            )

            print(
                "Current robot position:",
                tuple(
                    round(float(value), 4)
                    for value in current_position
                ),
            )

            print(
                "Planner target:",
                tuple(
                    round(float(value), 4)
                    for value in planner_target
                ),
            )

            print(
                "Robot target:",
                tuple(
                    round(float(value), 4)
                    for value in robot_target
                ),
            )


            # ------------------------------------------------
            # DISTANCE IN ACTUAL ROBOT SPACE
            # ------------------------------------------------

            distance = float(
                np.linalg.norm(
                    current_position
                    - robot_target
                )
            )


            print(
                f"Required movement: "
                f"{distance:.6f} m"
            )


            # ------------------------------------------------
            # ONLY SKIP IF ALREADY REALLY AT TARGET
            # ------------------------------------------------

            POSITION_TOLERANCE = 0.01


            if distance <= POSITION_TOLERANCE:

                print(
                    f"{object_name} is already "
                    "at the requested target."
                )

                continue


            # ------------------------------------------------
            # CREATE NEW PICK ACTION
            #
            # IMPORTANT:
            # Pick from the object's CURRENT real position,
            # not from the old planner initial position.
            # ------------------------------------------------

            arm = pick_action.arm

            result.append(
                Action(
                    action="PICK",
                    object_name=object_name,
                    arm=arm,
                    target=tuple(
                        float(value)
                        for value in current_position
                    ),
                )
            )


            # ------------------------------------------------
            # CREATE NEW PLACE ACTION
            #
            # Keep planner-space target here.
            # recovery_executor.py will convert it to
            # SO-101 robot coordinates.
            # ------------------------------------------------

            result.append(
                Action(
                    action="PLACE",
                    object_name=object_name,
                    arm=arm,
                    target=tuple(
                        float(value)
                        for value in planner_target
                    ),
                )
            )


        return result
