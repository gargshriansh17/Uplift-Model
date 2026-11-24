"""Run the uplift project pipeline stage by stage.

Usage (from the project root):
    python main.py --list                 # show all stages
    python main.py data model             # run selected stages in order
    python main.py all                    # run the full two-arm pipeline
    python main.py final-test             # one-time test evaluation (run LAST)

Every stage is a module run with `python -m`, so each one can also be run on
its own, e.g. `python -m src.models.lightgbm_t_learner`.
"""

import argparse
import subprocess
import sys

STAGES = {
    # --- data ---
    "eda": "src.eda",
    "data": "src.data_preprocessing",
    # --- two-arm models (Men's Email vs No Email) ---
    "baseline": "src.baseline",
    "t-learner": "src.models.t_learner",
    "s-learner": "src.models.s_learner",
    "lightgbm": "src.models.lightgbm_t_learner",
    # --- model selection (validation + repeated out-of-fold only) ---
    "compare": "src.model_comparison",
    "bootstrap": "src.bootstrap_evaluation",
    "repeated-oof": "src.repeated_oof_evaluation",
    # --- three-arm extension ---
    "three-arm-data": "src.prepare_three_arm_data",
    "three-arm-t-learner": "src.models.three_arm_t_learner",
    "three-arm-lightgbm": "src.models.three_arm_lightgbm_t_learner",
    "three-arm-policy": "src.models.three_arm_policy_evaluation",
    "three-arm-ranking": "src.models.three_arm_uplift_ranking",
    # --- final, one-time test evaluation ---
    "final-test": "src.final_test_evaluation",
}

# Everything needed to reproduce the model selection, in order.
# final-test is intentionally NOT included: it should be run once, last.
ALL_STAGES = [
    "eda", "data", "baseline", "t-learner", "s-learner", "lightgbm",
    "compare", "bootstrap", "repeated-oof",
]


def run_stage(name):
    module = STAGES[name]
    print(f"\n{'=' * 60}\nSTAGE: {name}  ({module})\n{'=' * 60}")
    result = subprocess.run([sys.executable, "-m", module])
    if result.returncode != 0:
        print(f"\nStage '{name}' failed with exit code {result.returncode}.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Uplift modelling pipeline")
    parser.add_argument("stages", nargs="*", help="stage names, or 'all'")
    parser.add_argument("--list", action="store_true", help="list stages")
    args = parser.parse_args()

    if args.list or not args.stages:
        print("Available stages:")
        for name, module in STAGES.items():
            print(f"  {name:<22} {module}")
        print("\n'all' runs:", ", ".join(ALL_STAGES))
        return

    requested = ALL_STAGES if args.stages == ["all"] else args.stages
    unknown = [s for s in requested if s not in STAGES]
    if unknown:
        parser.error(f"unknown stage(s): {', '.join(unknown)}")

    for name in requested:
        run_stage(name)


if __name__ == "__main__":
    main()
