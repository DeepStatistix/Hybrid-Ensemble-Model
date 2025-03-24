# Hybrid Ensemble Model (HEM)

This repository contains an implementation of the **Hybrid Ensemble Model (HEM)** for stock price forecasting. HEM combines four well-known ensemble learning techniques—**Bagging**, **Boosting**, **Stacking**, and **Dagging**—within a **meta-learning** framework. By leveraging each method’s complementary strengths, HEM aims to provide more accurate and robust forecasts than standalone models.

---

## Project Structure


1. **`data/`**  
   - **`raw/`**: Contains the original CSV file (e.g., `meta_stock_data.csv`)  
   - **`processed/`**: For any cleaned or feature-engineered data

2. **`notebooks/`**  
   - **`Melding2 (1).ipynb`**: Core notebook showing data exploration, model training, and evaluations

3. **`scripts/`**  
   - **`data_preprocessing.py`**: Functions to load data, drop missing values, create lag features, and split into train/test sets  
   - **`ensemble_methods.py`**: Unified code for training Bagging, Boosting, Stacking, and Dagging models  
   - **`hem.py`**: Hybrid Ensemble Model that aggregates sub-model predictions via a meta-learner  
   - **`evaluation.py`**: Basic metrics (MSE, R², sMAPE, Directional Accuracy) and statistical tests (Diebold-Mariano, Modified DM, MCS)  
   - **`main.py`**: High-level pipeline to run the entire workflow from data loading to final evaluation

4. **`tests/`**  
   - **`test_ensemble_methods.py`**: Automated tests to ensure each ensemble method works as expected

5. **`docs/`** (optional)  
   - **`your_paper.pdf`**: Paper or additional project documentation

---

## Installation

1. **Clone** this repository:
   ```bash
   git clone https://github.com/YourUsername/YourHEMProject.git
   cd YourHEMProject
