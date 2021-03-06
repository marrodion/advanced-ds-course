import time
import warnings
from collections import defaultdict

import fire
import joblib
import pandas as pd
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.linear_model import LogisticRegressionCV
import numpy as np

from fraud import preprocessing

SPLIT_PATH = preprocessing.DATA_PATH / "splits"
METRICS = ["accuracy", "f1_weighted", "precision", "recall", "roc_auc"]
SEED = 42
EXPERIMENTS_DIR = preprocessing.DATA_PATH / "experiments"
warnings.filterwarnings(action="ignore")


def cv_defined_folds(
    pipeline, metrics, train_path=SPLIT_PATH / "train", test_path=SPLIT_PATH / "test"
):
    n_folds = train_path.glob("fold_*.csv")
    metrics = defaultdict(dict)
    for i in range(n_folds):
        train_x, train_y = get_xy(pd.read_csv(train_path / f"fold_{i}.csv"))
        test_x, test_y = get_xy(pd.read_csv(test_path / f"fold_{i}.csv"))
        pipeline.fit(train_x, train_y)
        prediction = pipeline.predict_proba(test_x)
        for m in metrics():
            metrics[i][m.__name__] = m(test_y, prediction)
        return metrics


def cv(
    pipeline, metrics, data_path=preprocessing.RAW_FILE, cv=TimeSeriesSplit, n_splits=5
):
    df = preprocessing.load_data(path=data_path)
    x, y = get_xy(df)
    kfold = cv(n_splits=n_splits)
    results = cross_validate(
        pipeline,
        x,
        y,
        scoring=metrics,
        cv=kfold,
        n_jobs=-1,
        return_train_score=True,
        return_estimator=True,
    )
    return results


def get_xy(df, target_column=preprocessing.TARGET):
    y = df[target_column].values
    x = df.drop(target_column, axis=1)
    return x, y


def evaluate_and_save(est, data_path, out_path):
    cv_results = cv(est, METRICS, data_path=Path(data_path))
    estimators = cv_results.pop("estimator", {})
    metrics = pd.DataFrame.from_dict(cv_results).mean(axis=0)
    if out_path:
        out_path = Path(out_path)
        out_path.mkdir(parents=True, exist_ok=True)
        if estimators:
            joblib.dump(estimators, out_path / "estimators.pkl")
        if metrics.values.any():
            metrics.to_json(out_path / "metrics.json", orient="columns")
    return metrics.to_dict()


def train():
    est = LogisticRegressionCV(
        Cs=np.linspace(1e-3, 10, 10), cv=TimeSeriesSplit(n_splits=5), scoring='f1_weighted'
    )
    data_path = preprocessing.RAW_FILE
    out_path = EXPERIMENTS_DIR
    return evaluate_and_save(est, data_path=data_path, out_path=out_path)


if __name__ == "__main__":
    fire.Fire({"train": train})
