from pathlib import Path
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import fire

DATA_PATH = Path(__file__).parent.parent.parent / "data" 
RAW_FILE = DATA_PATH / "creditcard.csv"
TARGET = "Class"


def load_data(path=RAW_FILE):
    df = pd.read_csv(path)
    return df


def split_folds(in_path=RAW_FILE, out_dir=DATA_PATH.parent, n_splits=5):
    # data
    out_dir = Path(out_dir)
    (out_dir / 'train').mkdir(parents=True, exist_ok=True)
    (out_dir / 'test').mkdir(parents=True, exist_ok=True)
    df = load_data(in_path)
    y = df[TARGET]
    tss = TimeSeriesSplit(n_splits=5)
    for i, (train_idx, test_idx) in enumerate(tss.split(df, y)):
        df.iloc[train_idx].to_csv(out_dir / "train" / f"fold_{i}.csv")
        df.iloc[test_idx].to_csv(out_dir / "test" / f"fold_{i}.csv")


if __name__ == "__main__":
    fire.Fire()
