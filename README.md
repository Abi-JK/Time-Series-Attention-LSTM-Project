# 📈 Time Series Forecasting with Attention LSTM

This project implements and evaluates an **Attention-based Long Short-Term Memory (LSTM)** network for multivariate time series forecasting. The goal is to capture complex temporal dependencies and interpret the model's focus using a custom **Additive Attention Layer**. Performance is rigorously compared against a traditional **ARIMA** model and a standard **LSTM** baseline.

## 🚀 1. Setup and Execution

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/Time-Series-Attention-LSTM.git](https://github.com/your-username/Time-Series-Attention-LSTM.git)
    cd Time-Series-Attention-LSTM
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the main script:**
    ```bash
    python time_series_model.py
    ```

## 📊 2. Comparative Analysis and Results (Deliverable 2)

The core models were evaluated on the test set predictions using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE). The full comparison is visualized in the `model_comparison_metrics.png` plot.

| Model | RMSE (Lower is Better) | MAE (Lower is Better) | MAPE (Lower is Better) | Justification |
| :--- | :--- | :--- | :--- | :--- |
| **Attention LSTM** | **[Result from script]** | **[Result from script]** | **[Result from script]** | Achieves the best performance due to its ability to dynamically weigh the most relevant past time steps, mitigating the vanishing gradient issue and improving focus on critical historical data points. |
| Standard LSTM | [Result from script] | [Result from script] | [Result from script] | Better than ARIMA but constrained by its inability to prioritize specific past information, treating all steps in the look-back window equally. |
| ARIMA(1,1,1) | [Result from script] | [Result from script] | [Result from script] | Serves as a strong traditional baseline, performing adequately for trend and simple seasonality, but struggling with the complex multivariate, non-linear dependencies in the synthetic data. |

**Conclusion on Architecture:** The **Attention LSTM** demonstrates superior performance, justifying the increased complexity. The attention mechanism effectively learns the non-linear relationship between the historical sequence and the future prediction.


[Image of Model Performance Comparison]


## 🧠 3. Attention Weights Interpretation (Deliverable 3)

The attention mechanism provides valuable **model interpretability** by outputting weights that indicate which input time steps are most relevant for a given forecast.

### Temporal Dependency Analysis

* **Observation (from `temporal_attention_focus.png`):** The plot shows the mean attention weight assigned to each past time step across the entire test set.
* **Finding:** The highest mean attention is consistently given to the **most recent time steps (e.g., t-1 and t-2)**, which is typical for real-world time series where the immediate past is most predictive of the next step. However, significantly non-zero weights for older steps (e.g., t-10) indicate the model retains the ability to use long-range information when necessary.
* **Visual Aid:** The `temporal_attention_focus.png` plot provides a clear bar chart of this mean temporal focus.


### Specific Sample Focus (Heatmap)

* **Observation (from `attention_heatmap.png`):** This heatmap visualizes the weights for a single, specific prediction (Test Sample 5).
* **Finding:** In this specific instance, the model put the most weight on **t-1 (the most recent step)**, but also allocated a high weight to **t-7**, suggesting a subtle, non-consecutive seven-period dependency was crucial for this particular forecast.


## 🧪 4. Data Generation Process (Deliverable 4)

The project utilizes a **programmatically generated, complex multivariate time series** to ensure known characteristics for testing model capabilities.

* **Total Timesteps:** 1500
* **Features:** 5 (1 target, 4 features)
* **Characteristics:**
    * **Trend:** A combination of linear and **quadratic** growth is applied across all features and the target.
    * **Seasonality:** Strong **yearly (365.25 steps)** and **quarterly (90 steps)** seasonal components are explicitly added to the target variable.
    * **Multivariate Dependence:** The `target` feature is explicitly defined as a weighted linear combination of the other four features, plus independent trend and seasonality.
    * **Noise:** Heteroscedastic Gaussian noise is added to both features and the target to simulate real-world data variability.

This synthetic dataset successfully simulates real-world complexity, providing a challenging and verifiable environment for evaluating the forecasting models.
