"""
run all 4 experiments one after another
"""

import subprocess
import sys
from pathlib import Path

# experiments to run
experiments = [
    ("bilinear", "true"),
    ("bilinear", "false"),
    ("transposed", "true"),
    ("transposed", "false"),
]

print("="*60)
print("Running all 4 experiments")
print("="*60)

for i, (upsample, skip) in enumerate(experiments, 1):
    print(f"\n{'='*60}")
    print(f"Experiment {i}/4: upsample={upsample}, skip={skip}")
    print(f"{'='*60}")
    
    # run main.py with arguments
    cmd = [
        sys.executable,
        "scripts/main.py",
        "--upsample", upsample,
        "--skip", skip
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"Experiment failed! Stopping.")
        break

print("\n" + "="*60)
print("All experiments completed!")
print("="*60)