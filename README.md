# 🚛 Tariffly — Freight Tariff Rate Prediction

A client-side machine learning web application that predicts freight tariff rates (shipping costs) for truckload dispatches based on trip parameters prior to route departure.

Built with **React**, **Vite**, **scikit-learn**, and **Vitest**. Powered by an in-browser tree ensemble inference engine that runs 100% locally with zero backend API latency.

---

## 🌟 Key Features

- **In-Browser Client-Side ML Execution**: Model training occurs offline in Python, exporting the trained ensemble tree structure to JSON (`model.json`). The web UI executes predictions directly in JavaScript (`predict.js`) without requiring a backend server.
- **Target Leakage Protection**: Identified an exact formula leak in the raw dataset ($\text{tariff} = \text{labour} + \text{toll} + \text{miscellaneous}$). Direct post-hoc line-item expenses were removed from the feature set to ensure honest, leak-free forecasting from pre-dispatch trip characteristics.
- **5-Fold Cross-Validation Benchmark**: Evaluated 11 candidate regressors on the training set using scikit-learn pipelines (`ColumnTransformer`, median numeric imputer, mode categorical imputer, `StandardScaler`, `OneHotEncoder`).
- **Winning Model (Extra Trees Regressor)**: Selected **Extra Trees Regressor** as the top model, achieving a cross-validation RMSE of **1717.86** and a held-out test $R^2$ of **0.9303**.
- **Zero-Error Python-to-JS Parity**: Browser predictions match Python predictions to float precision (0.0 numerical error), verified via automated Vitest suites (`npm test`).

---

## 📊 Dataset & Features

The dataset (`tarifftruck1.csv`) contains 289 historical freight trip records across 11 features:

| Feature | Type | Kind | Description | Status |
| :--- | :---: | :---: | :--- | :---: |
| `year` | Integer | Numerical | Calendar year of trip (2013 – 2015) | **Used** |
| `road` | String | Categorical | Highway type (`nh`: National Highway, `sh`: State Highway) | **Used** |
| `seasonalimpact` | String | Categorical | Seasonal weather impact flag (`yes`, `no`) | **Used** |
| `demand` | String | Categorical | Freight market demand rating (`high`, `normal`, `less`, `low`) | **Used** |
| `wttons` | Float | Numerical | Cargo load weight in metric tonnes | **Used** |
| `distance` | Float | Numerical | Route distance in kilometers | **Used** |
| `fuelcost` | Float | Numerical | Incurred or estimated fuel expenditure | **Used** |
| `labour` | Float | Numerical | Driver & loading labor cost component | *Excluded (Leakage)* |
| `toll` | Float | Numerical | Route toll charges | *Excluded (Leakage)* |
| `miscellaneous` | Float | Numerical | Other miscellaneous route expenses | *Excluded (Leakage)* |
| `tariff` | Float | Numerical | **Target Variable**: Freight tariff rate charged | **Target** |

---

## 📈 Model Benchmark Comparison

Evaluated on training set ($N = 231$) using 5-fold cross-validation:

| Model | Mean CV RMSE | Mean CV MAE | Mean CV $R^2$ | Held-Out Test $R^2$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Extra Trees** | **1717.86** | **910.90** | **0.8487** | **0.9303** | **Selected Model** |
| **Gradient Boosting** | 1756.40 | 1021.36 | 0.8467 | 0.9015 | Candidate |
| **Random Forest** | 1807.76 | 1029.36 | 0.8369 | 0.8950 | Candidate |
| **HistGradientBoosting** | 1852.39 | 1072.42 | 0.8289 | 0.8841 | Candidate |
| **Decision Tree** | 1900.56 | 1131.17 | 0.8180 | 0.8620 | Candidate |
| **DummyRegressor (Mean)** | 4613.34 | 3593.78 | -0.0683 | -0.0120 | Baseline |
| **Linear / Ridge / Lasso** | 8956+ | 2389+ | -7.20 to -8.10 | — | Failed (Non-linear) |

---

## 📂 Project Architecture

```text
Prediction-of-Tariff-Rates-master/
├── Freight_Tariff_Rate_Prediction_Technical_Report.pdf  # Comprehensive PDF technical audit
├── README.md                                             # Documentation (you are here)
├── tarifftruck1.csv                                      # Historical trip dataset
└── frontend/                                             # Web Application & ML Pipeline
    ├── public/
    │   └── model.json                                    # Active ML model & preprocessor artifact
    ├── scripts/
    │   └── export_model.py                               # Training, 5-fold CV & tree export script
    ├── src/
    │   ├── App.jsx                                       # React user interface
    │   ├── predict.js                                    # In-browser JS inference engine
    │   └── predict.test.js                               # Vitest unit test suite
    ├── package.json                                      # Node project dependencies & scripts
    └── vite.config.js                                    # Vite bundler config
```

---

## 🚀 Getting Started

### 1. Requirements
- **Python**: `3.9+` (with `pandas`, `scikit-learn`, `numpy`)
- **Node.js**: `18+` (with `npm`)

### 2. Run Model Training & Export
To re-train candidate models, run 5-fold cross-validation, and export `model.json`:

```bash
python frontend/scripts/export_model.py
```

### 3. Start Frontend Development Server
To launch the React user interface:

```bash
cd frontend
npm run dev
```

Open [http://localhost:5173/](http://localhost:5173/) in your browser.

### 4. Run Unit Tests & Verify Model Parity
To verify JavaScript prediction parity against Python:

```bash
cd frontend
npm test
```

### 5. Build for Production
To generate a static production bundle:

```bash
cd frontend
npm run build
```

---

## 📄 Documentation & PDF Report

A complete 16-step architectural technical report is available in the root directory:
- [**Freight_Tariff_Rate_Prediction_Technical_Report.pdf**](./Freight_Tariff_Rate_Prediction_Technical_Report.pdf)
