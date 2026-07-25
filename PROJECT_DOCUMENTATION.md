# Smart Building Occupancy Detection — Project Documentation

## 1. Project Overview

This project develops a binary classification solution for smart-building occupancy detection.

Environmental sensor readings are used to determine whether a room is occupied or not occupied.

The solution contains two connected components:

1. A machine-learning component that performs occupancy classification.
2. An LLM component that explains the classification output and generates smart-building recommendations.

## 2. Problem Definition

The project addresses the following question:

> Can environmental sensor readings be used to determine whether a room is occupied?

The problem is formulated as a binary classification task.

### Inputs

- Temperature
- Light
- CO2
- HumidityRatio

### Output

- `0`: Not Occupied
- `1`: Occupied

## 3. Smart-Building Application

Accurate occupancy detection can support:

- Automatic lighting control
- HVAC optimization
- Ventilation management
- Energy-efficiency improvement
- Indoor environmental monitoring
- Smart-building decision support

## 4. Dataset Description

The dataset contains environmental sensor readings and a binary occupancy label.

It is provided as:

- A training file containing 8,143 observations.
- A testing file containing 9,752 observations.

The testing file remains separate throughout training, model comparison, and threshold selection.

## 5. Original Dataset Features

| Feature | Description |
|-----------------------
| date | Date and time of the reading |
| Temperature | Room temperature |
| Humidity | Relative humidity |
| Light | Light intensity |
| CO2 | Carbon dioxide concentration |
| HumidityRatio | Humidity ratio |
| Occupancy | Binary target variable |

## 6. Data-Quality Analysis

The following checks were performed:

- Dataset shape
- Column names
- Data types
- Missing values
- Duplicated rows
- Target distribution
- Numerical correlations
- Feature distributions

The results showed:

- No missing values.
- No duplicated rows.
- All selected model features were already numerical.
- The target contained two classes.

The `Unnamed: 0` column was removed because it was only a row identifier.

## 7. Target Distribution

The training dataset contains approximately:

- 79% Not Occupied
- 21% Occupied

This represents moderate class imbalance.

Validation and testing data were not manually balanced because they should preserve the original data distribution.

The models use balanced class weights during training.

## 8. Exploratory Data Analysis

### 8.1 Correlation Analysis

The correlation matrix was used to study relationships between sensor features and occupancy.

Main findings:

- Light had the strongest relationship with Occupancy.
- CO2 had a strong relationship with Occupancy.
- Temperature had a moderate relationship with Occupancy.
- HumidityRatio provided some predictive information.
- Humidity had the weakest relationship with Occupancy.

Humidity and HumidityRatio were highly correlated, indicating that they contained similar information.

### 8.2 Boxplot Analysis

The boxplots showed:

- Light values were generally higher for occupied observations.
- CO2 levels were generally higher when the room was occupied.
- The median temperature was higher for occupied observations.
- Humidity distributions had substantial overlap.
- HumidityRatio showed some separation between the classes.

Outliers were retained because they may represent valid sensor conditions rather than data errors.

## 9. Feature Selection

The final selected features were:

```python
features = [
    "Temperature",
    "Light",
    "CO2",
    "HumidityRatio"
]
```

The target was:

```python
target = "Occupancy"
```

### Excluded Features

#### Unnamed: 0

Removed because it was only a row identifier.

#### Humidity

Excluded because it was highly correlated with HumidityRatio and had a weaker relationship with Occupancy.

#### Date

Excluded because the final workflow used a stratified random split and focused on environmental sensor measurements.

## 10. Data Splitting

The provided training file was divided into:

- 80% training
- 20% validation

The provided testing file remained independent.

The split used:

```python
X_train, X_validation, y_train, y_validation = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Split Decisions

- `test_size=0.20` reserves 20% for validation.
- `random_state=42` makes the split reproducible.
- `stratify=y` preserves the original target-class proportions.

Stratification does not make the classes equal.

## 11. Preprocessing Pipelines

Each classifier was placed inside a Pipeline.

### Logistic Regression

```text
Input Features
→ StandardScaler
→ Logistic Regression
```

### Support Vector Machine

```text
Input Features
→ StandardScaler
→ SVM
```

### Random Forest

```text
Input Features
→ Passthrough
→ Random Forest
```

### Decision Tree

```text
Input Features
→ Passthrough
→ Decision Tree
```

Scaling was included inside the pipelines to prevent preprocessing leakage.

## 12. Model Training

Four models were trained:

- Logistic Regression
- Support Vector Machine
- Random Forest
- Decision Tree

All models received the same training and validation data.

Each model generated:

- Validation class predictions
- Validation probabilities for the occupied class

## 13. Evaluation Metrics

### Accuracy

The percentage of all correctly classified observations.

### Precision

The proportion of occupied predictions that were correct.

### Recall

The proportion of actual occupied observations detected by the model.

### F1-score

The balance between precision and recall.

### ROC-AUC

Measures the ability to distinguish between occupied and not-occupied observations across multiple thresholds.

### Average Precision

Summarizes the Precision–Recall relationship and is useful for evaluating the less frequent occupied class.

### Confusion Matrix

The confusion matrix presents:

- True negatives
- False positives
- False negatives
- True positives

## 14. Threshold Analysis

The default threshold for binary classification is 0.50.

The project evaluated thresholds from 0.10 to 0.90.

For each threshold:

1. Probabilities were converted into class predictions.
2. Accuracy was calculated.
3. Precision was calculated.
4. Recall was calculated.
5. F1-score was calculated.

The threshold with the highest validation F1-score was selected.

Test data was not used during threshold selection.

## 15. Model Comparison

Random Forest achieved the strongest validation performance.

### Selected Validation Results

| Metric | Result |
|---|---:|
| Threshold | 0.65 |
| Accuracy | 0.9963 |
| Precision | 0.9857 |
| Recall | 0.9971 |
| F1-score | 0.9914 |
| ROC-AUC | 0.9993 |
| Average Precision | 0.9954 |

These results indicate that Random Forest achieved strong classification performance while detecting almost all occupied observations in validation data.

## 16. Independent Test Evaluation

The selected Random Forest model was evaluated using the independent test dataset.

The model achieved:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.9924 |
| Average Precision | 0.9686 |

The independent test results show that the model maintained strong class-separation ability on unseen observations.

## 17. LLM Integration

Gemini was added after the machine-learning classification stage.

The LLM receives:

- Sensor readings
- Machine-learning prediction
- The probability generated by the classifier

The LLM then produces an explanation or smart-building recommendation.

The machine-learning model remains responsible for the occupancy classification.

### 17.1 Zero-Shot Prompting

The LLM receives the current sensor case without previous examples.

It generates:

- A short occupancy explanation
- An interpretation of the sensor conditions
- A simple building-management recommendation

### 17.2 Few-Shot Prompting

The LLM receives example sensor cases and expected control actions.

It then produces a structured recommendation for:

- Lights
- HVAC mode
- Ventilation

### 17.3 Reasoning Prompting

The LLM receives a scenario containing conflicting sensor signals.

It is asked to:

1. Identify the conflicting evidence.
2. Explain possible interpretations.
3. Assess the operational risk.
4. Recommend a cautious building action.

The explanations represent possible interpretations rather than confirmed physical events.

## 18. Final Workflow

```text
Load Data
    ↓
Inspect and Clean Data
    ↓
Analyze Target and Features
    ↓
Select Model Inputs
    ↓
Create Training and Validation Sets
    ↓
Build Model Pipelines
    ↓
Train Four Models
    ↓
Generate Validation Probabilities
    ↓
Evaluate Multiple Thresholds
    ↓
Select Random Forest
    ↓
Evaluate Independent Test Data
    ↓
Generate LLM Explanations and Recommendations
```

## 19. Limitations

- Sensor readings are sequential and nearby observations may be similar.
- Random splitting may make validation results slightly optimistic.
- Lighting behavior may differ between buildings.
- The dataset represents a specific environment.
- LLM responses may vary between executions.
- LLM outputs are recommendations, not additional model predictions.
- Real building automation requires operational testing and safeguards.

## 20. Future Work

Future improvements may include:

- Time-aware validation.
- Cross-validation.
- Probability calibration.
- Feature-importance analysis.
- Testing in multiple buildings.
- Real-time sensor integration.
- A live monitoring dashboard.
- Automated alerts and energy-management rules.

## 21. Conclusion

The project demonstrates a complete smart-building classification workflow.

Random Forest achieved the strongest occupancy-detection performance, while Gemini provided an explanatory layer for interpreting model outputs and generating operational recommendations.

The machine-learning classifier remains the source of the occupancy decision, while the LLM supports communication and decision interpretation.