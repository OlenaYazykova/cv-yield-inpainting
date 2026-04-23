import random
from pathlib import Path


def split_fields(
    data_root: str,
    seed: int = 42,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
) -> tuple[list[str], list[str], list[str]]:
    """
    Unified field split function for train/val/test.
    Returns (train_fields, val_fields, test_fields).
    If test_ratio=0.0, test_fields will be an empty list.
    """
    fields = sorted([p.name for p in Path(data_root).iterdir() if p.is_dir()])

    if len(fields) < 2:
        raise ValueError("At least 2 fields required.")

    rng = random.Random(seed)
    rng.shuffle(fields)

    n_total = len(fields)
    n_test = max(0, int(n_total * test_ratio))
    n_val = max(1, int(n_total * val_ratio))

    if n_test + n_val >= n_total:
        n_test = min(n_test, 1)
        n_val = 1

    test_fields = fields[:n_test]
    val_fields = fields[n_test:n_test + n_val]
    train_fields = fields[n_test + n_val:]

    if len(train_fields) == 0:
        raise ValueError("train_fields is empty after split.")

    return train_fields, val_fields, test_fields
