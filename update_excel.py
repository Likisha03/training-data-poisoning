
from pathlib import Path
import re

import pandas as pd
from sklearn.metrics import roc_auc_score

RESULTS = Path("results")

SPECTRAL_DIR = RESULTS / "spectral"
ACTIVATION_DIR = RESULTS / "activation"

SPECTRAL_CSV = SPECTRAL_DIR / "spectral_results.csv"
ACTIVATION_CSV = ACTIVATION_DIR / "activation_results.csv"

SPECTRAL_XLSX = SPECTRAL_DIR / "spectral_results.xlsx"
ACTIVATION_XLSX = ACTIVATION_DIR / "activation_results.xlsx"

DETAIL_COLUMNS = [
    "Algorithm",
    "dataset",
    "sample_id",
    "label",
    "suspicion_score",
    "cluster",
    "is_flagged",
    "is_poisoned",
]


def as_bool(series):
    return series.astype(str).str.strip().str.lower().isin(
        ["true", "1", "yes"]
    )


def load_results(path, algorithm):
    if not path.exists():
        print(f"Skipping {algorithm}: CSV not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    required = {
        "dataset",
        "sample_id",
        "label",
        "is_flagged",
        "is_poisoned",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    df["Algorithm"] = algorithm
    df["dataset"] = df["dataset"].astype(str).str.strip()
    df["_actual"] = as_bool(df["is_poisoned"])
    df["_predicted"] = as_bool(df["is_flagged"])

    if "suspicion_score" in df.columns:
        df["_score"] = pd.to_numeric(
            df["suspicion_score"], errors="coerce"
        )
    else:
        df["_score"] = float("nan")

    return df


def evaluate(group, algorithm, dataset):
    actual = group["_actual"]
    predicted = group["_predicted"]

    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    tn = int((~actual & ~predicted).sum())

    total = tp + fp + fn + tn

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0

    valid = group["_score"].notna()
    auc = float("nan")

    if valid.any() and group.loc[valid, "_actual"].nunique() == 2:
        auc = float(
            roc_auc_score(
                group.loc[valid, "_actual"].astype(int),
                group.loc[valid, "_score"],
            )
        )

    return {
        "Algorithm": algorithm,
        "Dataset": dataset,
        "Total Samples": total,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Specificity": specificity,
        "False Positive Rate": fpr,
        "False Negative Rate": fnr,
        "ROC-AUC": auc,
    }


def safe_sheet_name(name, used):
    clean = re.sub(r"[\\/*?:\[\]]", "_", str(name)).strip("'")
    clean = clean or "Dataset"
    clean = clean[:31]

    candidate = clean
    counter = 1

    while candidate in used:
        suffix = f"_{counter}"
        candidate = clean[: 31 - len(suffix)] + suffix
        counter += 1

    used.add(candidate)
    return candidate


def format_sheet(ws):
    ws.freeze_panes = "A2"

    if ws.max_row and ws.max_column:
        ws.auto_filter.ref = ws.dimensions

    for column in ws.columns:
        width = max(
            (
                len(str(cell.value))
                if cell.value is not None
                else 0
            )
            for cell in column
        )
        ws.column_dimensions[
            column[0].column_letter
        ].width = min(max(width + 2, 10), 45)

    headers = {
        cell.value: cell.column
        for cell in ws[1]
    }

    percentage_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "Specificity",
        "False Positive Rate",
        "False Negative Rate",
    ]

    for header in percentage_columns:
        if header in headers:
            column_number = headers[header]

            for row_number in range(2, ws.max_row + 1):
                ws.cell(
                    row_number, column_number
                ).number_format = "0.00%"

    if "ROC-AUC" in headers:
        column_number = headers["ROC-AUC"]

        for row_number in range(2, ws.max_row + 1):
            ws.cell(
                row_number, column_number
            ).number_format = "0.0000"


def build_report(csv_path, excel_path, algorithm):
    data = load_results(csv_path, algorithm)

    if data.empty:
        print(f"No results available for {algorithm}; workbook not created.")
        return

    metrics_rows = []
    confusion_rows = []

    for dataset, group in data.groupby("dataset", sort=True):
        metrics_rows.append(
            evaluate(group, algorithm, dataset)
        )

        actual = group["_actual"]
        predicted = group["_predicted"]

        confusion_rows.append({
            "Algorithm": algorithm,
            "Dataset": dataset,
            "TP": int((actual & predicted).sum()),
            "FP": int((~actual & predicted).sum()),
            "FN": int((actual & ~predicted).sum()),
            "TN": int((~actual & ~predicted).sum()),
        })

    metrics = pd.DataFrame(metrics_rows)
    confusion = pd.DataFrame(confusion_rows)

    datasets = sorted(data["dataset"].unique())

    summary = pd.DataFrame([
        ["Algorithm", algorithm],
        ["Datasets included", ", ".join(datasets)],
        ["Number of datasets", len(datasets)],
        ["Total result rows", len(data)],
        ["Source CSV", str(csv_path)],
        ["Excel output", str(excel_path)],
    ], columns=["Item", "Value"])

    if algorithm == "Spectral Signatures":
        thresholds = pd.DataFrame([
            ["MNIST", "train.csv", 24.061039110544318, 0.95, 50],
            ["CIFAR-10", "train_batch_1.csv", 203.0505151413023, 0.95, 50],
            ["CIFAR-10", "train_batch_2.csv", 207.11356228776577, 0.95, 50],
            ["CIFAR-10", "train_batch_3.csv", 209.66405487056616, 0.95, 50],
            ["CIFAR-10", "train_batch_4.csv", 200.87356449089333, 0.95, 50],
            ["CIFAR-10", "train_batch_5.csv", 200.09724400006644, 0.95, 50],
        ], columns=[
            "Dataset",
            "Input Run",
            "Threshold Value",
            "Threshold Quantile",
            "Power Iterations",
        ])
    else:
        thresholds = pd.DataFrame(columns=[
            "Dataset",
            "Input Run",
            "Threshold Value",
            "Threshold Quantile",
            "Power Iterations",
        ])

    preprocessing = pd.DataFrame([
        ["Algorithm", algorithm],
        ["Input CSV", str(csv_path)],
        ["Ground-truth column", "is_poisoned"],
        ["Prediction column", "is_flagged"],
        [
            "Continuous score column",
            "suspicion_score"
            if "suspicion_score" in data.columns
            else "Not available",
        ],
        ["Datasets in source", ", ".join(datasets)],
    ], columns=["Item", "Details"])

    details = data.copy()

    for column in DETAIL_COLUMNS:
        if column not in details.columns:
            details[column] = pd.NA

    details = details[DETAIL_COLUMNS]

    sheets = {
        "Executive Summary": summary,
        "Metrics Summary": metrics,
        "Confusion Matrix": confusion,
        "Threshold Details": thresholds,
        "Preprocessing": preprocessing,
    }

    used_sheet_names = set(sheets)

    # Add dedicated sheets for the standard datasets if present.
    for preferred in ("MNIST", "CIFAR-10"):
        subset = details[
            details["dataset"].str.upper() == preferred.upper()
        ]

        if not subset.empty:
            sheet_name = safe_sheet_name(
                preferred, used_sheet_names
            )
            sheets[sheet_name] = subset

    # Add a separate sheet for every other dataset found in the CSV.
    for dataset in datasets:
        if dataset.upper() in {"MNIST", "CIFAR-10"}:
            continue

        sheet_name = safe_sheet_name(
            dataset, used_sheet_names
        )

        sheets[sheet_name] = details[
            details["dataset"] == dataset
        ]

    excel_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(
        excel_path, engine="openpyxl"
    ) as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )
            format_sheet(writer.sheets[sheet_name])

    print(f"\n=== {algorithm.upper()} METRICS ===")
    print(metrics.round(4).to_string(index=False))
    print(f"Excel updated: {excel_path.resolve()}")


def main():
    build_report(
        SPECTRAL_CSV,
        SPECTRAL_XLSX,
        "Spectral Signatures",
    )

    build_report(
        ACTIVATION_CSV,
        ACTIVATION_XLSX,
        "Activation Clustering",
    )


if __name__ == "__main__":
    main()
