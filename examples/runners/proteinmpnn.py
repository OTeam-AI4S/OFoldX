from __future__ import annotations

from pathlib import Path

from ofoldx.runners import Runner

STRUCTURE = Path(__file__).resolve().parents[1] / "data" / "tripeptide.pdb"


def main() -> None:
    runner = Runner(
        {
            "steps": 1,
            "model": {"architecture": "proteinmpnn"},
            "data": {"train": str(STRUCTURE)},
            "optim": {"type": "adamw", "lr": 1e-4},
            "dataloader": {"batch_size": 1, "num_workers": 0},
            "logging": {"enabled": False},
            "workspace": {"root": "runs/proteinmpnn"},
            "ckpt": {"enabled": False},
        }
    )
    try:
        runner.train()
    finally:
        runner.close()


if __name__ == "__main__":
    main()
