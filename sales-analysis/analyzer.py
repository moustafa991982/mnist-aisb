# %% Imports
from pathlib import Path
import json

import pandas as pd


# %% Paths
SCRIPT_DIR = Path(__file__).resolve().parent      # .../sales -analysis
PARENT_DIR = SCRIPT_DIR.parent                    # one level higher: .../mnist-aisb
DATA_DIR = SCRIPT_DIR / "data"                    # .../sales -analysis/data
print(f"Looking for Excel files in {DATA_DIR}")


# %% Convert a CSV file to Excel
def csv_to_excel(csv_path, excel_path=None):
    """Convert a CSV file to an Excel file (sales.csv -> sales.xlsx, same folder)."""
    csv_path = Path(csv_path)
    if excel_path is None:
        excel_path = csv_path.with_suffix(".xlsx")

    df = pd.read_csv(csv_path)
    df.to_excel(excel_path, index=False)   # index=False: don't add a 0,1,2... column

    print(f"Saved {len(df)} row(s) from {csv_path.name} -> {excel_path}")
    return Path(excel_path)


# %% Convert an Excel file to JSON
def excel_to_json(excel_path, json_path=None):
    """Read every sheet of an Excel file and save it as JSON.

    By default the JSON file is written one level above this script's folder,
    named after the Excel file (sales.xlsx -> sales.json).
    """
    excel_path = Path(excel_path)
    print(f"Reading {excel_path.name}...")
    if json_path is None:
        json_path = PARENT_DIR / f"{excel_path.stem}.json"
    json_path = Path(json_path)
    print(f"Saving JSON to {json_path.name}...")
    # sheet_name=None -> dict of {sheet name: DataFrame}
    sheets = pd.read_excel(excel_path, sheet_name=None)

    # Each sheet becomes a list of row records: [{"col": value, ...}, ...]
    # Round-trip through to_json so dates/NaN become JSON-safe values.
    result = {
        name: json.loads(df.to_json(orient="records", date_format="iso"))
        for name, df in sheets.items()
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(result)} sheet(s) from {excel_path.name} -> {json_path}")
    return json_path


# %% Run
if __name__ == "__main__":
    # Convert every Excel file found in the data folder
    print(f"Looking for Excel files in {DATA_DIR}")

    csv_to_excel(DATA_DIR / "sales.csv")

    print(f"Looking for Excel files in {DATA_DIR}")
    excel_files = sorted(DATA_DIR.glob("*.xls*"))
    if not excel_files:
        print(f"No Excel files (.xlsx / .xls) found in {DATA_DIR}")
        print("Copy your Excel file into that folder and run again.")
    for excel_file in excel_files:
        if excel_file.name.startswith("~$"):   # skip Excel's temporary lock files
            continue
        excel_to_json(excel_file)

# %%
