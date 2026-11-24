#This file is for managing paths and common project settings

from pathlib import Path
Root_DIR=Path(__file__).resolve().parents[1]

DATA_DIR=Root_DIR/"data"
RAW_DATA_DIR=DATA_DIR/"raw"
PROCESSED_DATA_DIR=DATA_DIR/'processed'

ARTIFACTS_DIR=Root_DIR/"artifacts" #artifacts are the files generated during the model development
REPORTS_DIR=Root_DIR/"reports"#This will tell us where are the analysis results saved
FIGURES_DIR=REPORTS_DIR/"figures"

DATA_PATH=RAW_DATA_DIR/"hillstrom.csv"

SEGMENT_COL="segment"
TARGET_COL="conversion"
TREATMENT_SEGMENT="Mens E-Mail"
CONTROL_SEGMENT="No E-Mail"
RANDOM_STATE=42
TEST_SIZE=0.20
VALIDATION_SIZE=0.20

for directory in [
    PROCESSED_DATA_DIR,
    ARTIFACTS_DIR,
    REPORTS_DIR,
    FIGURES_DIR
]:

    directory.mkdir(parents=True,exist_ok=True)

print("Project configuration loaded successfully")

