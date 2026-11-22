"""Feature-only, line-oriented prediction worker. No feedback or model updates."""
import argparse
import json
import sys

from .cli import prediction
from .data import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    model = load_model(args.checkpoint)

    def guard(event, args):
        # Accidental Python I/O is blocked after imports/checkpoint loading.
        # This supplements, rather than replaces, platform mount isolation.
        if event == "open" or event.startswith("socket.") or event == "subprocess.Popen":
            raise RuntimeError("Feature-only prediction forbids file/network/subprocess access after loading")
    sys.addaudithook(guard)
    for line in sys.stdin:
        request = json.loads(line)
        if (not isinstance(request, dict) or set(request) != {"task", "inputs"}
                or request["task"] != "classification"):
            raise ValueError("Prediction accepts only task and inputs; labels and IDs are forbidden")
        outputs = prediction(model, request["inputs"])
        print(json.dumps({"outputs": outputs}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
