"""Input and output locations for the operative spine cohort."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
V2 = True
AFFILIATIONS = ROOT / "data/affiliations.csv"
PANEL = ROOT / "data/mips_panel.csv"
FRAME = ROOT / "data/spine_services.csv"
COHORT = ROOT / "cohort"
RESULTS = ROOT / "results"
EXPANSION = RESULTS / "inputs"
