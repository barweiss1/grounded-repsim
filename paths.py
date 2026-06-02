import os
from pathlib import Path

# Override this on servers with:
#   export SIM_METRIC_RESOURCES=/path/to/sim_metric_resources
#
# By default, use a sim_metric_resources directory next to this repository.
resources_path = Path(
    os.environ.get(
        "SIM_METRIC_RESOURCES",
        Path(__file__).resolve().parent / "sim_metric_resources",
    )
)
