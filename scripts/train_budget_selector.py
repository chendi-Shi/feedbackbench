"""Train nine small CPU budget selectors from completed, task-disjoint episodes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from embodied_research.budget_learning import TrainingConfig, train_campaign  # noqa: E402


def main(argv=None) -> None:
    from embodied_research.runtime import configure_platform
    configure_platform()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Training collection, never the test directory")
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--risk-scales", type=float, nargs=3, default=(0.05, 0.05, 0.1),
                        help="Normalized action proxy scales, not physical safety tolerances")
    parser.add_argument("--residual-quantile", type=float, default=0.9)
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=TrainingConfig().thresholds)
    args = parser.parse_args(argv)
    config = TrainingConfig(epochs=args.epochs, batch_size=args.batch_size,
                            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                            risk_scales=tuple(args.risk_scales), residual_quantile=args.residual_quantile,
                            thresholds=tuple(args.thresholds))
    result = train_campaign(args.data, args.output, config=config, project_root=ROOT)
    print(json.dumps({"status": result["status"], "models": len(result["models"]),
                      "summary": str(args.output.resolve() / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
