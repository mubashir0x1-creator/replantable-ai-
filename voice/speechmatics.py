import os
import time
import requests


API_URL = "https://asr.api.speechmatics.com/v2"


def transcribe_audio(audio_file):
    api_key = os.getenv("SPEECHMATICS_API_KEY")

    if not api_key:
        raise RuntimeError("SPEECHMATICS_API_KEY is not set.")

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    config = {
        "type": "transcription",
        "transcription_config": {
            "language": "en"
        }
    }

    with open(audio_file, "rb") as f:
        files = {
            "data_file": f
        }

        data = {
            "config": str(config).replace("'", '"')
        }

        response = requests.post(
            f"{API_URL}/jobs/",
            headers=headers,
            files=files,
            data=data,
        )

    response.raise_for_status()

    job = response.json()

    job_id = job["id"]

    print(f"Speechmatics job created: {job_id}")

    while True:
        response = requests.get(
            f"{API_URL}/jobs/{job_id}",
            headers=headers,
        )

        response.raise_for_status()

        status = response.json()

        job_status = status.get("job", {}).get(
            "status",
            status.get("status")
        )

        print(f"Job status: {job_status}")

        if job_status == "done":
            break

        if job_status in ["rejected", "failed"]:
            raise RuntimeError(
                f"Speechmatics job failed: {status}"
            )

        time.sleep(2)

    response = requests.get(
        f"{API_URL}/jobs/{job_id}/transcript",
        headers=headers,
        params={"format": "txt"},
    )

    response.raise_for_status()

    return response.text


if __name__ == "__main__":
    print("Speechmatics voice module ready.")
