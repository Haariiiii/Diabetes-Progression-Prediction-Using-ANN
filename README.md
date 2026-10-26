# Diabetes Progression Prediction Using ANN

## Overview

This project uses an Artificial Neural Network (ANN) built with TensorFlow/Keras to predict diabetes disease progression using Scikit-learn's built-in Diabetes dataset. It is a **regression** task because the target is a continuous numerical value.

## Dataset

The dataset is loaded directly from `sklearn.datasets.load_diabetes`.

- **Samples:** 442
- **Input features:** 10
- **Target:** A quantitative measure of disease progression one year after baseline
- **Missing values:** Checked during preprocessing; none were found in the loaded dataset.

The features describe baseline variables such as age, sex, body mass index (BMI), blood pressure, and six blood-serum measurements.

## Workflow

1. **Load the data** using Scikit-learn.
2. **Inspect and prepare the data** by separating the input features (`X`) and target (`y`) and checking for missing values.
3. **Explore the data** using feature histograms, a correlation heatmap, and feature skewness.
4. **Split the data** into training and testing sets using an 80/20 split with `random_state=42`.
5. **Scale the features** using `StandardScaler`. The scaler is fitted on the training data and then applied to the test data to avoid data leakage.
6. **Build an ANN** with hidden layers using ReLU activation and one linear output neuron.
7. **Train the model** using Adam, Mean Squared Error (MSE) loss, batch size 16, validation data, and early stopping.
8. **Evaluate performance** using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.
9. **Experiment with model architecture** and compare results.

## Model Architecture

The best-performing architecture reported in the notebook is:

```text
Input layer: 10 features
Hidden layer 1: 16 neurons, ReLU
Hidden layer 2: 8 neurons, ReLU
Output layer: 1 neuron, Linear
```

### Training Configuration

- **Optimizer:** Adam
- **Loss function:** Mean Squared Error (`mse`)
- **Additional metric:** Mean Absolute Error (`mae`)
- **Maximum epochs:** 300
- **Batch size:** 16
- **Early stopping:** Monitors validation loss and restores the best weights
- **Train/test split:** 80% / 20%

## Results

The best R² score recorded in the notebook is:

| Metric | Result |
|---|---:|
| R² score | **0.4626** |

An R² score of 0.4626 means the model explains approximately 46.26% of the variation in the target values on the evaluation split.

MSE and MAE are also calculated by the notebook. Add their final values to this table after running the final version:

| Metric | Result |
|---|---:|
| Mean Squared Error (MSE) | Add final notebook result |
| Mean Absolute Error (MAE) | Add final notebook result |

## Requirements

Install the required Python packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
```

## How to Run

1. Clone or download this repository.
2. Open `Deeplearning_Asessment.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
3. Run the cells in order.
4. Review the visualizations, training history, and final regression metrics.

## Project Structure

```text
.
└── Deeplearning_Asessment.ipynb
```

## Notes

- This project predicts a **continuous measure of diabetes progression**; it does not classify whether a person has diabetes.
- The dataset is small, so scores can vary between train/test splits.
- A higher R² and lower MSE/MAE indicate better predictive performance.
- Use validation results to guide model selection and keep the test set for final evaluation.
- This is an educational machine-learning project, not a medical diagnostic tool.

## Tools and Technologies

- Python
- NumPy and Pandas
- Matplotlib and Seaborn
- Scikit-learn
- TensorFlow / Keras
- Jupyter Notebook / Google Colab
