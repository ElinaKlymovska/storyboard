"""Allow video_shots to be run as a module."""

import sys
from video_shots.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
