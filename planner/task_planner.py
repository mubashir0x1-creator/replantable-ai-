# RePlanTable AI
#
# Natural-language task planner.
#
# AI assistance: ChatGPT was used to help draft
# the initial task-planning structure.


from dataclasses import dataclass
import re


# ------------------------------------------------------------
# Action representation
# ------------------------------------------------------------

@dataclass
class Action:
    action: str
    object_name: str
    arm: str
    target: tuple


# ------------------------------------------------------------
# Known objects
# ------------------------------------------------------------

OBJECT_ARMS = {
    "plate": "left",
    "cup": "right",
}


# ------------------------------------------------------------
# Object synonyms
# ------------------------------------------------------------

OBJECT_ALIASES = {
    "plate": [
        "plate",
        "dish",
    ],

    "cup": [
        "cup",
        "mug",
    ],
}


# ------------------------------------------------------------
# Target positions
# ------------------------------------------------------------

TARGETS = {
    "plate": {
        "initial": (-0.45, 0.0, 1.08),
        "center": (0.00, 0.0, 1.08),
        "left": (-0.15, 0.0, 1.08),
    },

    "cup": {
        "initial": (0.45, 0.0, 1.20),
        "left": (-0.05, 0.0, 1.20),
        "center": (0.00, 0.0, 1.20),
        "right": (0.15, 0.0, 1.20),
    },
}


# ------------------------------------------------------------
# Target synonyms
# ------------------------------------------------------------

TARGET_ALIASES = {

    "center": [
        "center",
        "middle",
        "centered",
    ],

    "left": [
        "left",
        "left side",
    ],

    "right": [
        "right",
        "right side",
    ],
}


# ------------------------------------------------------------
# Task Planner
# ------------------------------------------------------------

class TaskPlanner:

    def __init__(self):

        self.current_plan = []


    # --------------------------------------------------------
    # Create a pick + place action pair
    # --------------------------------------------------------

    def add_move(
        self,
        object_name,
        target_name,
    ):

        if object_name not in OBJECT_ARMS:

            raise ValueError(
                f"Unknown object: {object_name}"
            )

        if target_name not in TARGETS[object_name]:

            raise ValueError(
                f"Unknown target '{target_name}' "
                f"for {object_name}"
            )

        arm = OBJECT_ARMS[
            object_name
        ]

        pick_target = TARGETS[
            object_name
        ]["initial"]

        place_target = TARGETS[
            object_name
        ][target_name]

        self.current_plan.append(
            Action(
                action="pick",
                object_name=object_name,
                arm=arm,
                target=pick_target,
            )
        )

        self.current_plan.append(
            Action(
                action="place",
                object_name=object_name,
                arm=arm,
                target=place_target,
            )
        )


    # --------------------------------------------------------
    # Normalize object name
    # --------------------------------------------------------

    def find_object(self, text):

        for object_name, aliases in (
            OBJECT_ALIASES.items()
        ):

            for alias in aliases:

                pattern = (
                    r"\b"
                    + re.escape(alias)
                    + r"\b"
                )

                if re.search(
                    pattern,
                    text,
                ):

                    return object_name

        return None


    # --------------------------------------------------------
    # Find target inside text
    # --------------------------------------------------------

    def find_target(self, text):

        for target_name, aliases in (
            TARGET_ALIASES.items()
        ):

            for alias in aliases:

                pattern = (
                    r"\b"
                    + re.escape(alias)
                    + r"\b"
                )

                if re.search(
                    pattern,
                    text,
                ):

                    return target_name

        return None


    # --------------------------------------------------------
    # Parse object -> target relationships
    # --------------------------------------------------------

    def extract_object_targets(
        self,
        text,
    ):
        """
        Find object/target pairs while keeping
        the relationship between them.

        Example:

        "move the plate to the middle and
         put the cup on the right"

        becomes:

        plate -> center
        cup   -> right
        """

        results = []

        object_patterns = []

        for object_name, aliases in (
            OBJECT_ALIASES.items()
        ):

            for alias in aliases:

                object_patterns.append(
                    (
                        object_name,
                        alias,
                    )
                )


        matches = []

        for object_name, alias in (
            object_patterns
        ):

            pattern = (
                r"\b"
                + re.escape(alias)
                + r"\b"
            )

            for match in re.finditer(
                pattern,
                text,
            ):

                matches.append(
                    (
                        match.start(),
                        match.end(),
                        object_name,
                    )
                )


        matches.sort(
            key=lambda item: item[0]
        )


        for index, (
            start,
            end,
            object_name,
        ) in enumerate(matches):

            if index + 1 < len(matches):

                next_start = (
                    matches[index + 1][0]
                )

            else:

                next_start = len(text)


            clause = text[
                start:next_start
            ]


            target = self.find_target(
                clause
            )


            if target is None:

                continue


            results.append(
                (
                    object_name,
                    target,
                    start,
                )
            )


        return results


    # --------------------------------------------------------
    # Parse natural-language instruction
    # --------------------------------------------------------

    def parse_instruction(
        self,
        instruction,
    ):

        if not instruction:

            self.current_plan = []

            return []


        text = instruction.lower().strip()


        # Replace punctuation with spaces.
        text = re.sub(
            r"[,.!?;:]",
            " ",
            text,
        )


        self.current_plan = []


        pairs = self.extract_object_targets(
            text
        )


        # ----------------------------------------------------
        # Remove duplicate object instructions.
        #
        # The latest instruction for an object wins.
        # ----------------------------------------------------

        object_targets = {}

        object_order = []

        for (
            object_name,
            target_name,
            position,
        ) in pairs:

            if object_name not in object_targets:

                object_order.append(
                    object_name
                )

            object_targets[
                object_name
            ] = target_name


        # ----------------------------------------------------
        # Create planner actions.
        # ----------------------------------------------------

        for object_name in object_order:

            target_name = object_targets[
                object_name
            ]

            self.add_move(
                object_name,
                target_name,
            )


        return self.current_plan


    # --------------------------------------------------------
    # Default demonstration plan
    # --------------------------------------------------------

    def create_plan(self):

        self.current_plan = []


        self.add_move(
            "plate",
            "center",
        )


        self.add_move(
            "cup",
            "right",
        )


        return self.current_plan


    # --------------------------------------------------------
    # Display plan
    # --------------------------------------------------------

    def show_plan(self):

        if not self.current_plan:

            print()
            print(
                "No actions generated."
            )

            return


        print()
        print("=" * 64)
        print("TASK PLAN")
        print("=" * 64)


        for number, action in enumerate(
            self.current_plan,
            start=1,
        ):

            print(
                f"{number}. "
                f"{action.action.upper()} "
                f"{action.object_name} "
                f"WITH {action.arm.upper()} ARM "
                f"AT {action.target}"
            )


        print("=" * 64)


# ------------------------------------------------------------
# Test natural-language planning
# ------------------------------------------------------------

def main():

    print()
    print("=" * 64)
    print("RePlanTable AI")
    print("Natural Language Task Planner Test")
    print("=" * 64)


    planner = TaskPlanner()


    test_instructions = [

        "Put the plate in the center "
        "and the cup on the right.",

        "Move the plate to the middle "
        "and put the cup on the left.",

        "Keep the plate centered "
        "and move the cup right.",

        "Plate in the center, "
        "cup on the left.",

        "Place the mug on the right "
        "and move the dish to the middle.",
    ]


    for instruction in test_instructions:

        print()
        print("USER INSTRUCTION:")
        print(instruction)


        planner.parse_instruction(
            instruction
        )


        planner.show_plan()


if __name__ == "__main__":

    main()
