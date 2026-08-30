"""Run the TorchScript audio AI worker as ``python -m audio_ai``."""

from .worker import main


if __name__ == "__main__":
    raise SystemExit(main())
