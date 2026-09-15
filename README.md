# RePlanTable AI

## AI Voice + Vision-Aware Continuous Robot Replanning

RePlanTable AI is an intelligent bimanual robotic manipulation system that
understands natural-language voice instructions, plans robot actions,
executes them in MuJoCo, verifies the result using robot state and vision,
and automatically recovers from failures.

The system maintains a world state and uses the actual state of the scene
for continuous, state-aware replanning.

## How It Works

AI Voice
↓
Speechmatics
↓
Task Planner
↓
State-Aware Replanning
↓
Dual SO-101 Robot Execution
↓
Robot Verification
↓
Vision Verification
↓
Failure Detection
↓
Automatic Recovery
↓
Updated World State

## Main Features

- AI voice instruction processing
- Natural-language task planning
- Dual-arm SO-101 robot manipulation
- Pick and place actions
- World-state tracking
- Robot-state verification
- Vision-based verification
- Failure detection
- Automatic recovery
- State-aware replanning
- Continuous multi-instruction interaction

## Example

Voice instruction:

> "Move the cup to the left."

The system converts this instruction into a robot task:

1. Pick the cup with the right arm
2. Move the cup to the requested position
3. Release the cup
4. Verify the final position

## Failure Recovery

The system deliberately injects a placement failure during the demo.

The robot and vision verification systems detect that the cup did not reach
its intended target.

Instead of stopping, RePlanTable AI:

1. Detects the failed object
2. Reads its actual current position
3. Generates a recovery plan from that actual state
4. Moves the robot to the failed object
5. Grasps the object
6. Moves it to the target
7. Releases it
8. Verifies the result again

In the demonstrated recovery, the failed cup was recovered successfully.
The robot verification error was approximately 0.0347 m and the vision
verification also passed.

## Continuous Replanning

After recovery, the system continues processing new voice instructions
without restarting the simulation.

Example session:

1. "Move the cup to the left."
2. Forced failure occurs
3. Automatic recovery succeeds
4. "Move the plate to the center."
5. State-aware replanning executes the new task
6. "Move the cup to the right."
7. State-aware replanning uses the cup's current position
8. Session completes successfully

## Technology Stack

- Python
- MuJoCo
- NumPy
- Speechmatics
- Requests
- Analytical Inverse Kinematics
- Robot State Tracking
- Vision Verification
- State-Aware Replanning

## Project Structure

```text
ai-infra-hackathon/
├── planner/
│   ├── kinematics.py
│   ├── task_planner.py
│   └── replanner.py
├── simulation/
│   ├── dual_arm.py
│   ├── dual_arm_ik.py
│   ├── coordinate_adapter.py
│   ├── bimanual_manipulation.py
│   ├── recovery_executor.py
│   └── run_planned_task.py
├── state/
│   └── world_state.py
├── vision/
│   ├── live_detector.py
│   ├── verifier.py
│   └── vision_pipeline.py
├── voice/
│   ├── speechmatics.py
│   └── voice_input.py
└── ui/
