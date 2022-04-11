from pathlib import Path

# Path to compiled starspace binary in docker container
BINARY_PATH = Path("/StarSpace")

ARTIFACT_PATH = Path("../artifacts/marketplace")

# Params for starspace model: https://github.com/facebookresearch/Starspace
STARSPACE_PARAMS = {"trainMode": 1, "epoch": 100, "validationPatience": 10, "dim": 100}
