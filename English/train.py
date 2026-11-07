"""English train entry point; run from any directory."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fs2.train import main
if __name__ == "__main__":
    main(Path(__file__).with_name("config.yaml"))
