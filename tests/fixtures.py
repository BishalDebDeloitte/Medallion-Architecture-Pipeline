"""Sample data shared by agent tests."""

from pathlib import Path

import pandas as pd

from core.paths import bronze_file, silver_file


def write_csv(directory: Path, name: str, frame: pd.DataFrame) -> str:
    """Write a CSV and return its path as a string."""
    path = directory / name
    frame.to_csv(path, index=False)
    return str(path)


def make_orders() -> pd.DataFrame:
    """31 order rows: 30 distinct plus one exact duplicate; 2 statuses; mostly-null discount."""
    rows = [
        {
            "order_id": i,
            "customer_id": (i % 5) + 1,
            "amount": i * 10.5,
            "status": "open" if i % 2 else "closed",
            "discount": 5.0 if i <= 10 else None,
            "region": " north" if i % 3 == 0 else "south",
            "order_date": f"2024-01-{i:02d}",
        }
        for i in range(1, 31)
    ]
    rows.append(rows[-1])
    return pd.DataFrame(rows)


def make_customers() -> pd.DataFrame:
    """Five customers with unique ids."""
    return pd.DataFrame({"customer_id": [1, 2, 3, 4, 5], "name": list("abcde")})


def write_bronze(run_id: str) -> list[str]:
    """Write Bronze-style Parquet files for orders and customers under the run-scoped names."""
    paths = []
    for table, frame in (("orders", make_orders()), ("customers", make_customers())):
        path = bronze_file(table, run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        paths.append(str(path))
    return paths


def write_silver(run_id: str) -> list[str]:
    """Write Silver-style Parquet files (with injected pk columns) for orders and customers."""
    paths = []
    for table, frame in (("orders", make_orders().drop_duplicates()), ("customers", make_customers())):
        frame = frame.reset_index(drop=True)
        frame.insert(0, f"pk_{table}_silver_id", range(1, len(frame) + 1))
        path = silver_file(table, run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        paths.append(str(path))
    return paths
