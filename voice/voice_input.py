from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from voice.speechmatics import transcribe_audio


def get_voice_instruction(audio_file):
    audio_file = Path(audio_file)

    if not audio_file.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_file}"
        )

    transcript = transcribe_audio(
        str(audio_file)
    ).strip()

    if not transcript:
        raise RuntimeError(
            "Speechmatics returned an empty transcript."
        )

    return transcript


if __name__ == "__main__":

    audio_file = (
        Path(__file__).resolve().parent
        / "059e1dd5-99ee-4a26-a920-3ce06538d961.m4a"
    )

    instruction = get_voice_instruction(
        audio_file
    )

    print()
    print("=" * 64)
    print("VOICE TRANSCRIPT")
    print("=" * 64)
    print(instruction)
