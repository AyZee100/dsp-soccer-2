from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT_DIR / "data" / "source_data" / "input_data.csv"
LEGACY_INPUT_PATH = ROOT_DIR / "input_data.csv"
RAW_DATA_DIR = ROOT_DIR / "data" / "raw_data"


def split_csv(input_path: Path, output_folder: Path, num_files: int) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    df = pd.read_csv(input_path)
    chunks = np.array_split(df, num_files)
    output_folder.mkdir(parents=True, exist_ok=True)

    for i, chunk in enumerate(chunks):
        chunk.to_csv(output_folder / f"batch_{i}.csv", index=False)

    print(f"Created {num_files} files in {output_folder}")


if __name__ == "__main__":
    input_path = INPUT_PATH if INPUT_PATH.exists() else LEGACY_INPUT_PATH
    split_csv(input_path, RAW_DATA_DIR, 30)
