# Smart Building Occupancy 

A machine-learning and LLM-assisted project that predicts whether a room is occupied using environmental sensor readings. The project supports smart-building applications such as automatic lighting, HVAC control, ventilation management, and energy optimization.

## Project Objective

The objective is to build and evaluate a binary classification system that predicts room occupancy using sensor data.

The project also integrates Gemini to explain machine-learning outputs and generate smart-building recommendations.

## Dataset

The project uses the Occupancy Detection Dataset from Kaggle:

https://www.kaggle.com/datasets/saumitgp/occupancy-detection-dataset

The dataset includes:

- `DataTraining.csv`: used for training and validation.
- `DataTest.csv`: used for final independent evaluation.

## Target Variable

The target variable is `Occupancy`:

- `0`: Not Occupied
- `1`: Occupied

## Selected Features

The final model uses:

- Temperature
- Light
- CO2
- HumidityRatio

`Humidity` was excluded because it was highly correlated with `HumidityRatio` and had a weaker relationship with the target.

The raw `date` column was not used during model training.

## Models

Four classification models were trained and compared:

1. Logistic Regression
2. Support Vector Machine
3. Random Forest
4. Decision Tree

Logistic Regression and SVM use feature scaling inside their pipelines.

Random Forest and Decision Tree use the original feature values.

## Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision
- Classification Reports
- Confusion Matrices
- ROC Curves
- Precision–Recall Curves

Multiple classification thresholds between `0.10` and `0.90` were also evaluated using validation data.

## Main Results

Random Forest achieved the strongest validation performance.

| Metric | Validation Result  |
|--------|--------------------|
| Selected Threshold | 0.65 |
| Accuracy | 0.9963 |
| Precision | 0.9857 |
| Recall | 0.9971 |
| F1-score | 0.9914 |
| ROC-AUC | 0.9993 |
| Average Precision | 0.9954 |

Independent test performance:

| Metric  | Test Result |
|---------|-------------|
| ROC-AUC | 0.9924 |
| Average Precision | 0.9686 |

## LLM Integration

Gemini was integrated after the machine-learning classification stage.

The LLM does not replace the classifier. It receives the sensor values and machine-learning result, then produces an explanation or smart-building recommendation.

Three prompting approaches were demonstrated:

- Zero-shot prompting
- Few-shot prompting
- Reasoning prompting

## Project Structure

```text
Smart_Building/
├── data/
│   ├── DataTraining.csv
│   └── DataTest.csv
├── PROJECT_DOCUMENTATION.md
├── Smart_Building.ipynb
├── README.md
├── pyproject.toml
└── uv.lock
```

## Installation

Install the dependencies using uv:

```bash
uv sync
```

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

Open `Smart_Building.ipynb` in VS Code and select the Python interpreter from `.venv`.

Run the notebook cells from top to bottom.

## Documentation

Detailed project documentation is available in:

```text
PROJECT_DOCUMENTATION.md
```

## Limitations

- The sensor records are sequential, so nearby observations may be similar.
- The model may perform differently in buildings with different lighting and ventilation patterns.
- LLM outputs are explanatory recommendations and not additional classifier predictions.

## Future Work

- Time-aware validation
- Feature-importance analysis
- Testing in multiple buildings
- Real-time sensor integration
- Live occupancy dashboard

## Team Members

- Member 1: Joud Alrubaish 
- Member 2: Lama Alfreah 
- Member 3: Joud Alraimi
- Member 4: Fay Almasaud


Note : this project was developed for educational purposes a a part of AI Solutions Development Bootcamp - Tuwaiq Academy 
