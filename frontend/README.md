# Tariffly browser frontend

A React + Vite frontend for tariff estimates. Inference runs entirely in the
browser using a static JSON export of the notebook's fitted scikit-learn linear
regression pipeline. Trip inputs are not sent to a server.

## Run locally

From this directory:

```sh
npm install
npm run dev
```

## Refresh the browser model

From the project root, run:

```sh
python frontend/scripts/export_model.py
```

The exporter repeats the notebook's train/test split and cross-validation,
selects the best model by cross-validation RMSE, and writes
`frontend/public/model.json`. Browser inference currently supports the
linear-regression model; the exporter stops with an explicit error if another
estimator wins model selection.

## Checks

```sh
npm test
npm run lint
npm run build
```
