````markdown
# Opening the Black Box of Rainfall Prediction
### Regional Feature Importance in Deep Learning vs. Traditional Neural Networks

**Author:** Joseph LaMonica  
**Course:** CS444 Deep Learning - Final Project  
**Institution:** Emory University

## Project Overview
This project investigates the effectiveness and interpretability of Deep Learning models for meteorological forecasting, specifically predicting next-day rainfall in Australia.

While Deep Learning models like LSTMs and GRUs often outperform traditional methods, they are frequently treated as "black boxes." This study not only benchmarks **ANN, LSTM, and GRU** architectures against each other but also utilizes **SHAP (SHapley Additive exPlanations)** to interpret *what* the models are learning.

The key research questions addressed are:
1. Do complex Recurrent Neural Networks (RNNs) significantly outperform simpler Artificial Neural Networks (ANNs) for rainfall prediction?
2. Can we interpret these models to verify they are learning actual meteorological physics rather than just memorizing statistical artifacts?
3. How do feature importances differ between **Coastal** and **Inland** climate regions?

## Key Features
* **Deep Learning Models:** Implementation of three distinct architectures using **PyTorch**:
    * **ANN:** Baseline Feed-Forward Neural Network (flattened input).
    * **LSTM:** Long Short-Term Memory network for capturing temporal dependencies.
    * **GRU:** Gated Recurrent Unit for efficient temporal modeling.
* **Advanced Preprocessing:** * Circular encoding for cyclical features (Wind Direction, Months).
    * Weighted Binary Cross Entropy Loss to handle severe class imbalance (No Rain vs. Rain).
* **Explainable AI (XAI):** Integration of **DeepSHAP** to visualize feature importance and compare model decision-making processes against meteorological domain knowledge.
* **Regional Analysis:** Separate model training and evaluation for Global, Coastal (e.g., Sydney), and Inland (e.g., Alice Springs) datasets.

## 📂 Repository Structure
```text
.
├── data/
│   └── weatherAUS.csv       # Dataset (Rain in Australia)
├── scripts/
│   ├── analysis.py          # Exploratory Data Analysis (EDA) & missing value visualization
│   ├── explain.py           # SHAP analysis and feature importance plotting
│   ├── models.py            # PyTorch definitions for ANN, LSTM, and GRU classes
│   ├── preprocess.py        # Data loading, cleaning, and circular encoding
│   └── train.py             # Training loop, evaluation metrics, and model saving
├── analysis_plots/          # Generated plots from EDA
├── *.pth                    # Saved best model weights (e.g., Global_GRU_best.pth)
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies
````

## Installation & Requirements

This project requires Python 3.8+ and the following libraries:

```bash
pip install torch numpy pandas scikit-learn shap matplotlib seaborn
```

## Usage

### 1\. Data Analysis

Run the analysis script to generate statistical summaries and visualizations of the dataset (missing values, target distribution, etc.).

```bash
python scripts/analysis.py
```

### 2\. Training Models

Train the ANN, LSTM, and GRU models. You can toggle between training "Global" models or region-specific models inside the script.

```bash
python scripts/train.py
```

*Outputs: Saved model weights (`.pth` files) and performance metrics (Accuracy, F1-Score, ROC-AUC) printed to the console.*

### 3\. SHAP Interpretability

Generate SHAP summary plots to visualize which features drove the model predictions.

```bash
python scripts/explain.py
```

*Outputs: Feature importance plots saved as PNG files (e.g., `shap_Coastal_LSTM.png`).*

## Key Results

### Performance

  * **Best Overall Balance:** The **GRU** model offered the best balance of performance and efficiency. It achieved an F1-score slightly higher than the LSTM (0.6787 vs 0.6767) while training **\~19% faster** (315s vs 387s).
  * **ANN Limitations:** The simpler ANN struggled to capture temporal patterns, often "cheating" by memorizing static location data rather than learning weather dynamics.

### Interpretability (SHAP)

  * **Physical Validity:** Both LSTM and GRU models successfully learned to prioritize physical drivers of weather (Humidity, Pressure, Wind Speed) over static location identifiers.
  * **Regional Differences:**
      * **Coastal Models:** Heavily emphasized **Wind Gust Speed** and Seasonality (Month), reflecting the mechanical wind forces that drive coastal storms.
      * **Inland Models:** Prioritized **Pressure** systems (troughs), which aligns with meteorological literature on what drives rain in dry, inland regions.

## References

  * **Data Source:** [Rain in Australia (Kaggle)](https://www.google.com/search?q=https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)
  * **Methodology:**
      * *Lundberg & Lee (2017)* on SHAP (SHapley Additive exPlanations).
      * *Hochreiter & Schmidhuber (1997)* on Long Short-Term Memory (LSTM).

-----

*Created by Joseph LaMonica for CS444, Fall 2025.*

```
```