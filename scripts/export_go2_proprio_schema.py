"""Write the canonical Go2 proprioceptive observation/action schema JSON."""

from __future__ import annotations

import argparse
import json
import importlib
from pathlib import Path


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("output", type=Path)
  parser.add_argument(
    "--schema-module",
    default="src.tasks.velocity.config.go2.sim2real_schema",
  )
  args = parser.parse_args()
  schema = importlib.import_module(args.schema_module)
  payload = {
    "schema_sha256": schema.schema_sha256(),
    **schema.schema_payload(),
  }
  serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
  main()
