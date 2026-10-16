# CryoGuard

CryoGuard is a prototype cold-storage monitoring application. It combines a
FastAPI service, a browser-based dashboard, and Python modules for telemetry
feature processing and thermal-model experimentation.

The project is for demonstration and research. It has not been validated for
clinical, regulatory, or production use and must not be used as the sole basis
for vaccine or medicine storage decisions.

## Features

- FastAPI endpoints for health checks, predictions, replay scenarios, and
  explanations.
- Static dashboard pages served by the API.
- Python feature engineering, data processing, and digital-twin code.
- Small model-comparison summaries in `cryoguard/data/`.

## Run locally

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn cryoguard.backend.main:app --reload
```

Open <http://127.0.0.1:8000/>. The dashboard loads some styles and charts from
CDNs, so those assets require an internet connection.

## Data

The raw synthetic refrigeration datasets are maintained separately at
<https://github.com/iben0x/fridge-sensor-data>. They are licensed under
CC BY 4.0 by Islam Benamirouche; follow the upstream attribution and citation
requirements when using the data.

The processed CSV files are intentionally not included here: together they
occupy multiple gigabytes and can be recreated from the raw data. From the
repository root, clone the dataset and run the processing module:

```powershell
git clone https://github.com/iben0x/fridge-sensor-data.git
python -m cryoguard.src.data_processing.process_all_data
```

Generated files are written to `cryoguard/data/` and are ignored by Git. The
small comparison-summary CSVs in that directory are retained in the repository.

## Project layout

- `cryoguard/backend/` — FastAPI application.
- `cryoguard/frontend/` — dashboard pages served by the backend.
- `cryoguard/src/` — data processing, features, training, and physics modules.
- `cryoguard/docs/` — project progress and implementation notes.
- `src/` — root-level compatibility wrappers for selected physics modules.

## Licensing

No license has been specified for the CryoGuard application code. The dataset
has a separate CC BY 4.0 license; that license does not apply to this
application code.
