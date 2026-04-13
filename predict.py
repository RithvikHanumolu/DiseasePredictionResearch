from __future__ import annotations

from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


MODEL_PATH = Path("survey_models_comparison.pkl")
AUGMENTED_DATA_PATH = Path("augmented_diabetes_data.csv")
REAL_DATA_PATH = Path("cleaned_diabetes_survey.csv")
FALLBACK_DATA_PATH = Path("Diabetes.csv")
TARGET_CANDIDATES = ["true_diabetes_label", "label", "Diabetes_012", "target", "diabetes"]
FOLLOWUP_TEXT_CANDIDATES = ["FollowUpResponses"]
FOLLOWUP_QUESTION_PATTERN = re.compile(r"^Q\d+$", re.IGNORECASE)
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 42


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    return pd.read_csv(path, low_memory=False)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    canonical_columns = {
        "participant_id": "participant_id",
        "label": "label",
        "timestamp": "timestamp",
        "highbp": "HighBP",
        "highchol": "HighChol",
        "cholcheck": "CholCheck",
        "bmi": "BMI",
        "smoker": "Smoker",
        "stroke": "Stroke",
        "heartdiseaseorattack": "HeartDiseaseorAttack",
        "physactivity": "PhysActivity",
        "fruits": "Fruits",
        "veggies": "Veggies",
        "hvyalcoholconsump": "HvyAlcoholConsump",
        "anyhealthcare": "AnyHealthcare",
        "nodocbccost": "NoDocbcCost",
        "genhlth": "GenHlth",
        "menthlth": "MentHlth",
        "physhlth": "PhysHlth",
        "diffwalk": "DiffWalk",
        "sex": "Sex",
        "age": "Age",
        "education": "Education",
        "income": "Income",
        "q1": "Q1",
        "q2": "Q2",
        "q3": "Q3",
        "q4": "Q4",
        "q5": "Q5",
    }
    renamed = {column: canonical_columns.get(column.strip().lower(), column) for column in df.columns}
    return df.rename(columns=renamed)


def find_target_column(df: pd.DataFrame) -> str:
    for column in TARGET_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(f"Could not find a target column. Available columns include: {df.columns[:30].tolist()}")


def normalize_binary_target(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().all():
        lowered = series.astype(str).str.strip().str.lower()
        mapped = lowered.map(
            {
                "no": 0,
                "false": 0,
                "negative": 0,
                "non-diabetic": 0,
                "yes": 1,
                "true": 1,
                "positive": 1,
                "diabetic": 1,
                "diabetes": 1,
            }
        )
        numeric = pd.to_numeric(mapped, errors="coerce")

    numeric = numeric.where(numeric.isin([0, 1]), (numeric > 0).astype(float))
    return numeric


def get_followup_columns(df: pd.DataFrame) -> list[str]:
    explicit = [column for column in FOLLOWUP_TEXT_CANDIDATES if column in df.columns]
    question_cols = [column for column in df.columns if FOLLOWUP_QUESTION_PATTERN.match(column)]
    return explicit + sorted(question_cols)


def build_followup_text(row: pd.Series, followup_columns: list[str]) -> list[str]:
    answers: list[str] = []
    for column in followup_columns:
        value = row.get(column, "")
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none"}:
            answers.append(text)
    return answers


def contains_any(text: str, keywords: tuple[str, ...]) -> int:
    return int(any(keyword in text for keyword in keywords))


def engineer_followup_features(df: pd.DataFrame, followup_feature_names: list[str]) -> pd.DataFrame:
    followup_columns = get_followup_columns(df)
    features = pd.DataFrame(index=df.index)

    if not followup_columns:
        for feature in followup_feature_names:
            features[feature] = 0.0
        return features

    answers_per_row = df.apply(lambda row: build_followup_text(row, followup_columns), axis=1)
    joined = answers_per_row.apply(lambda answers: " ".join(answers).lower())

    features["fu_response_count"] = answers_per_row.apply(len).astype(float)
    features["fu_total_chars"] = answers_per_row.apply(lambda answers: float(sum(len(answer) for answer in answers)))
    features["fu_avg_chars"] = np.where(
        features["fu_response_count"] > 0,
        features["fu_total_chars"] / features["fu_response_count"],
        0.0,
    )
    features["fu_mentions_exercise"] = joined.apply(
        lambda text: contains_any(text, ("exercise", "workout", "walk", "walking", "run", "gym", "yoga"))
    )
    features["fu_mentions_diet"] = joined.apply(
        lambda text: contains_any(text, ("diet", "food", "meal", "eat", "nutrition", "vegetable", "fruit"))
    )
    features["fu_mentions_sleep"] = joined.apply(
        lambda text: contains_any(text, ("sleep", "insomnia", "rest", "tired", "bedtime"))
    )
    features["fu_mentions_stress"] = joined.apply(
        lambda text: contains_any(text, ("stress", "anxiety", "depress", "overwhelm", "mental"))
    )
    features["fu_mentions_smoke"] = joined.apply(
        lambda text: contains_any(text, ("smoke", "smoking", "cigarette", "nicotine", "vape"))
    )
    features["fu_mentions_alcohol"] = joined.apply(
        lambda text: contains_any(text, ("alcohol", "drink", "beer", "wine", "liquor"))
    )
    features["fu_mentions_family"] = joined.apply(
        lambda text: contains_any(text, ("family", "mother", "father", "parent", "sibling", "genetic"))
    )
    features["fu_mentions_medication"] = joined.apply(
        lambda text: contains_any(text, ("medication", "medicine", "metformin", "insulin", "pill", "drug"))
    )

    for feature in followup_feature_names:
        if feature not in features.columns:
            features[feature] = 0.0

    return features[followup_feature_names].astype(float)


def summarize_followup_transformation(df: pd.DataFrame, followup_feature_names: list[str], sample_rows: int = 3) -> None:
    followup_columns = get_followup_columns(df)
    transformed = engineer_followup_features(df, followup_feature_names)

    print()
    print("Follow-up transformation review:")
    print(f"Source columns: {followup_columns if followup_columns else 'None'}")
    print("Transformation steps:")
    print("1. Collect non-empty answers from FollowUpResponses and/or Q1..Qn columns.")
    print("2. Join answers into one lowercase text blob per row.")
    print("3. Derive response-count and character-length features.")
    print("4. Set keyword flags for exercise, diet, sleep, stress, smoke, alcohol, family, medication.")

    if transformed.empty:
        print("No transformed follow-up features available.")
        return

    print()
    print("Transformed follow-up feature coverage:")
    coverage = pd.DataFrame(
        {
            "nonzero_rows": (transformed > 0).sum(axis=0).astype(int),
            "mean_value": transformed.mean(axis=0).round(4),
        }
    )
    print(coverage.to_string())

    print()
    print("Sample row transformation audit:")
    sample_count = min(sample_rows, len(df))
    for idx in transformed.index[:sample_count]:
        raw_answers = build_followup_text(df.loc[idx], followup_columns)
        print(f"Row {idx}:")
        print(f"  Raw answers: {raw_answers if raw_answers else '[]'}")
        print(f"  Engineered features: {transformed.loc[idx].to_dict()}")


def coerce_numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(index=df.index)
    for column in columns:
        if column in df.columns:
            frame[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            frame[column] = 0.0
    return frame.fillna(0.0)


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "NPV": npv,
    }


def bootstrap_metric_intervals(
    y_true: pd.Series,
    baseline_pred: np.ndarray,
    baseline_prob: np.ndarray,
    personalized_pred: np.ndarray,
    personalized_prob: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, int]:
    rng = np.random.default_rng(seed)
    records: list[dict[str, float]] = []
    y_array = y_true.to_numpy()
    n = len(y_array)

    for _ in range(iterations):
        sample_idx = rng.integers(0, n, size=n)
        y_sample = y_array[sample_idx]
        if len(np.unique(y_sample)) < 2:
            continue

        base_sample = compute_metrics(
            pd.Series(y_sample),
            baseline_pred[sample_idx],
            baseline_prob[sample_idx],
        )
        personalized_sample = compute_metrics(
            pd.Series(y_sample),
            personalized_pred[sample_idx],
            personalized_prob[sample_idx],
        )

        record: dict[str, float] = {}
        for metric_name in base_sample:
            record[f"{metric_name}_baseline"] = base_sample[metric_name]
            record[f"{metric_name}_personalized"] = personalized_sample[metric_name]
            record[f"{metric_name}_delta"] = personalized_sample[metric_name] - base_sample[metric_name]
        records.append(record)

    if not records:
        return pd.DataFrame(), 0

    bootstrap_df = pd.DataFrame(records)
    summary_rows = []
    for metric_name in compute_metrics(
        y_true,
        baseline_pred,
        baseline_prob,
    ):
        delta_series = bootstrap_df[f"{metric_name}_delta"]
        summary_rows.append(
            {
                "Metric": metric_name,
                "Delta Mean": delta_series.mean(),
                "Delta 2.5%": delta_series.quantile(0.025),
                "Delta 97.5%": delta_series.quantile(0.975),
                "P(Personalized > Baseline)": (delta_series > 0).mean(),
            }
        )

    return pd.DataFrame(summary_rows), len(records)


def print_metric_table(base_metrics: dict[str, float], personalized_metrics: dict[str, float]) -> None:
    metric_names = list(base_metrics.keys())
    header = f"{'Metric':<20}{'Baseline':>12}{'Personalized':>16}{'Improvement':>14}"
    print(header)
    print("-" * len(header))
    for metric in metric_names:
        base_value = base_metrics[metric]
        personalized_value = personalized_metrics[metric]
        delta = personalized_value - base_value
        print(f"{metric:<20}{base_value:>12.4f}{personalized_value:>16.4f}{delta:>14.4f}")


def print_error_rows(
    df: pd.DataFrame,
    y_true: pd.Series,
    baseline_pred: np.ndarray,
    baseline_prob: np.ndarray,
    personalized_pred: np.ndarray,
    personalized_prob: np.ndarray,
) -> None:
    report = df.loc[y_true.index, ["participant_id", "label"]].copy() if "participant_id" in df.columns else pd.DataFrame(index=y_true.index)
    report["true_label"] = y_true.astype(int)
    report["baseline_pred"] = baseline_pred
    report["baseline_prob"] = baseline_prob
    report["personalized_pred"] = personalized_pred
    report["personalized_prob"] = personalized_prob

    baseline_wrong = report[report["baseline_pred"] != report["true_label"]]
    personalized_wrong = report[report["personalized_pred"] != report["true_label"]]

    print()
    print("Baseline wrong rows:")
    if baseline_wrong.empty:
        print("None")
    else:
        print(baseline_wrong.to_string(index=True))

    print()
    print("Personalized wrong rows:")
    if personalized_wrong.empty:
        print("None")
    else:
        print(personalized_wrong.to_string(index=True))


def print_bootstrap_summary(summary_df: pd.DataFrame, used_iterations: int, requested_iterations: int) -> None:
    print()
    print("Bootstrap uncertainty review:")
    print(f"Valid bootstrap resamples: {used_iterations}/{requested_iterations}")
    if summary_df.empty:
        print("Not enough class variation to estimate intervals.")
        return

    formatted = summary_df.copy()
    for column in ["Delta Mean", "Delta 2.5%", "Delta 97.5%", "P(Personalized > Baseline)"]:
        formatted[column] = formatted[column].map(lambda value: f"{value:.4f}")
    print(formatted.to_string(index=False))


def evaluate_dataset(
    df: pd.DataFrame,
    dataset_path: Path,
    section_title: str,
    baseline_model,
    personalized_model,
    base_features: list[str],
    followup_features: list[str],
) -> None:
    target_column = find_target_column(df)
    y = normalize_binary_target(df[target_column])
    valid_mask = y.notna()

    X_base = coerce_numeric_frame(df, base_features)
    X_followup = engineer_followup_features(df, followup_features)
    X_personalized = pd.concat([X_base, X_followup], axis=1)

    X_base = X_base.loc[valid_mask]
    X_personalized = X_personalized.loc[valid_mask]
    y = y.loc[valid_mask].astype(int)

    baseline_prob = baseline_model.predict_proba(X_base)[:, 1]
    baseline_pred = (baseline_prob >= 0.5).astype(int)

    personalized_prob = personalized_model.predict_proba(X_personalized)[:, 1]
    personalized_pred = (personalized_prob >= 0.5).astype(int)

    base_metrics = compute_metrics(y, baseline_pred, baseline_prob)
    personalized_metrics = compute_metrics(y, personalized_pred, personalized_prob)

    followup_activity = int((X_followup.loc[valid_mask, "fu_response_count"] > 0).sum())

    print()
    print("=" * 72)
    print(section_title)
    print("=" * 72)
    print(f"Dataset: {dataset_path}")
    print(f"Rows evaluated: {len(y)}")
    print(f"Target column: {target_column}")
    print(f"Rows with follow-up content: {followup_activity}")
    print()
    print_metric_table(base_metrics, personalized_metrics)

    if dataset_path == REAL_DATA_PATH:
        summarize_followup_transformation(df.loc[valid_mask], followup_features)
        print_error_rows(df, y, baseline_pred, baseline_prob, personalized_pred, personalized_prob)
        bootstrap_summary, used_iterations = bootstrap_metric_intervals(
            y,
            baseline_pred,
            baseline_prob,
            personalized_pred,
            personalized_prob,
        )
        print_bootstrap_summary(bootstrap_summary, used_iterations, BOOTSTRAP_ITERATIONS)


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    baseline_model = bundle["base_model"]
    personalized_model = bundle["base_plus_followup_model"]
    base_features = bundle["base_features"]
    followup_features = bundle["followup_features"]

    augmented_df = load_csv(AUGMENTED_DATA_PATH)
    evaluate_dataset(
        augmented_df,
        AUGMENTED_DATA_PATH,
        "AUGMENTED / SYNTHETIC EVALUATION",
        baseline_model,
        personalized_model,
        base_features,
        followup_features,
    )

    if REAL_DATA_PATH.exists():
        real_df = load_csv(REAL_DATA_PATH)
        real_df = standardize_columns(real_df)
        evaluate_dataset(
            real_df,
            REAL_DATA_PATH,
            "REAL RESPONSE EVALUATION",
            baseline_model,
            personalized_model,
            base_features,
            followup_features,
        )
    elif FALLBACK_DATA_PATH.exists():
        fallback_df = load_csv(FALLBACK_DATA_PATH)
        evaluate_dataset(
            fallback_df,
            FALLBACK_DATA_PATH,
            "FALLBACK EVALUATION",
            baseline_model,
            personalized_model,
            base_features,
            followup_features,
        )


if __name__ == "__main__":
    main()
