"""
Master Pipeline - Runs all analysis workflows
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
WORKFLOWS_PATH = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"

scripts = [
    "run_data_processing.py",
    "run_ml_training.py",
    "run_optimization.py",
    "run_hmm_analysis.py",
    "run_backtest.py",
    "run_outlier_analysis.py"
]

print("="*70)
print("RUNNING ALL ANALYSIS WORKFLOWS")
print("="*70)

for i, script in enumerate(scripts, 1):
    script_path = WORKFLOWS_PATH / script
    print(f"\n[{i}/{len(scripts)}] Running {script}...")
    print("-" * 70)
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(WORKFLOWS_PATH),
            capture_output=False
        )
        
        if result.returncode != 0:
            print(f"!!! Error running {script}. Pipeline stopped.")
            sys.exit(1)
        else:
            print(f" {script} completed successfully")
    
    except Exception as e:
        print(f" Error running {script}: {e}")

print("\n" + "="*70)
print(" ALL WORKFLOWS COMPLETED!")
print("="*70)
print(f"\nCheck the following directories for outputs:")
print(f"   {PROJECT_ROOT / 'results'}")
print(f"   {PROJECT_ROOT / 'plots'}")
print(f"   {PROJECT_ROOT / 'models'}")
