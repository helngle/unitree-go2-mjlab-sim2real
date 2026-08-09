"""Run the nine serial high-slope evaluations required for V8 selection."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ITEMS = (
  (
    "v7",
    ROOT / "logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt",
    "Unitree-Go2-Rough-V7",
  ),
  *tuple(
    (
      f"control_234_model_{update}",
      ROOT / f"logs/rsl_rl/go2_velocity/2026-08-04_11-39-43_go2_v8_privileged_lin_vel_teacher_control_234_2048env_400iter/model_{update}.pt",
      "Unitree-Go2-Rough-V8-PrivilegedLinVelTeacher-Control",
    )
    for update in (100, 200, 300, 400)
  ),
  *tuple(
    (
      f"candidate_237_model_{update}",
      ROOT / f"logs/rsl_rl/go2_velocity/2026-08-04_11-57-17_go2_v8_privileged_lin_vel_teacher_candidate_237_2048env_400iter/model_{update}.pt",
      "Unitree-Go2-Rough-V8-PrivilegedLinVelTeacher",
    )
    for update in (100, 200, 300, 400)
  ),
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
