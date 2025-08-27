import argparse
import json
import sys
from typing import Any, Dict

import numpy as np

from regret import regret_from_payoff_matrix


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def run_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    payoffs = payload["payoffs"]
    actions = payload["actions"]
    mode = payload.get("mode", "static")
    losses = bool(payload.get("losses", False))
    result = regret_from_payoff_matrix(payoffs, actions, mode=mode, losses=losses)
    return {k: _to_serializable(v) for k, v in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute regret from JSON input")
    parser.add_argument("--input", type=str, default="", help="Path to JSON input file")
    parser.add_argument("--stdin", action="store_true", help="Read JSON from stdin")
    args = parser.parse_args()

    if args.stdin:
        payload = json.load(sys.stdin)
    elif args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        print("Provide --stdin or --input <file>", file=sys.stderr)
        sys.exit(2)

    output = run_from_payload(payload)
    json.dump(output, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

