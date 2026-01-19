## `audits/v1/analysis_calibration/` (calibration selection metadata)

This directory contains the **grid-search calibration output** used to select checker parameters on the audited subset.

- **Primary artifact**: `audit_metrics.json`
  - `calibration.best_parameters`: selected \((m, t_{det}, \tau)\)
  - `calibration.best_metrics`: objective terms at the selected \(\tau\)

### Important note about curves / confusion matrices

For clarity, the audit analyses are split by evaluation variant:

- **Uncalibrated checker analysis** (risk--coverage + confusion): `audits/v1/analysis_uncalibrated/`
- **Calibrated checker analysis** (risk--coverage + confusion): `audits/v1/analysis_calibrated/`

The presence of `risk_coverage.png` here is historical and should not be treated as the calibrated curve.

