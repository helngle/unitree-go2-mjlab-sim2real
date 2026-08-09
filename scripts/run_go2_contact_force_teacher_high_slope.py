"""Run eight serial high-slope evaluations for contact-force Teacher selection."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
  "control_234": (
    ROOT / "logs/rsl_rl/go2_velocity/2026-08-10_01-05-52_go2_contact_force_teacher_v1_control_234_2048env_400iter",
    "Unitree-Go2-Rough-ContactForceTeacher-V1-Control",
  ),
  "candidate_246": (
    ROOT / "logs/rsl_rl/go2_velocity/2026-08-10_01-24-25_go2_contact_force_teacher_v1_candidate_246_2048env_400iter",
    "Unitree-Go2-Rough-ContactForceTeacher-V1",
  ),
}
ITEMS = tuple(
  (f"{arm}_model_{update}", run / f"model_{update}.pt", task_id)
  for update in (100, 200, 300, 400)
  for arm, (run, task_id) in RUNS.items()
)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--resume-existing", action="store_true")
  args = parser.parse_args()
  output_root = args.output_dir.expanduser().resolve()
  for label, checkpoint, task_id in ITEMS:
    output = output_root / label / "high_slope_matched.json"
    if output.exists():
      if args.resume_existing:
        print(f"SKIP {label}", flush=True)
        continue
      raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = (
      sys.executable,
      str(ROOT / "scripts/evaluate_go2_high_slope_matched.py"),
      "--checkpoint", str(checkpoint.resolve()),
      "--task-id", task_id,
      "--profiles", "clean", "randomized",
      "--radii", "2.5",
      "--output-file", str(output),
    )
    print(f"START {label}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"DONE {label}", flush=True)


if __name__ == "__main__":
  main()
