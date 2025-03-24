"""
evaluation.py

Contains:
1. Standard forecasting metrics (MSE, RMSE, R^2, sMAPE, Directional Accuracy).
2. Diebold-Mariano (DM) test.
3. Modified Diebold-Mariano (M-DM) test for small-sample bias.
4. Model Confidence Set (MCS) procedure.

If your Melding2 (1).ipynb has slightly different code for these tests,
copy/paste or adapt that code below so it matches exactly.
"""

import numpy as np
from math import sqrt
import statsmodels.api as sm

# -----------------------------------------------------------------
# 1. Standard Metrics
# -----------------------------------------------------------------
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def rmse(y_true, y_pred):
    return sqrt(mse(y_true, y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def smape(y_true, y_pred):
    return 100.0 * np.mean(
        2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))
    )

def directional_accuracy(y_true, y_pred):
    """
    Measures how often the model predicts the correct direction
    of change from one time step to the next.
    """
    correct = 0
    for i in range(1, len(y_true)):
        real_dir = np.sign(y_true[i] - y_true[i-1])
        pred_dir = np.sign(y_pred[i] - y_pred[i-1])
        if real_dir == pred_dir:
            correct += 1
    return correct / (len(y_true) - 1) * 100

# -----------------------------------------------------------------
# 2. Diebold-Mariano (DM) Test
# -----------------------------------------------------------------
def diebold_mariano_test(y_true, y_pred1, y_pred2, h=1, loss_function='SE'):
    """
    Performs the Diebold-Mariano test comparing two forecast models.

    :param y_true: Actual values (array-like).
    :param y_pred1: Predictions from Model 1 (array-like).
    :param y_pred2: Predictions from Model 2 (array-like).
    :param h: Forecast horizon (1 for one-step ahead).
    :param loss_function: 'SE' (squared error) or 'AE' (absolute error), etc.
    :return: (DM statistic, p-value)
    """
    # 1) Compute forecast errors
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2
    
    # 2) Compute the loss differential d_t
    if loss_function == 'SE':
        d = (e1 ** 2) - (e2 ** 2)
    elif loss_function == 'AE':
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("Unsupported loss function. Use 'SE' or 'AE' or adapt.")
    
    # 3) Mean of d
    d_bar = np.mean(d)
    
    # 4) Number of predictions
    T = len(d)

    # 5) Variance of d using Newey-West correction for serial correlation
    #   statsmodels provides convenient methods:
    #   https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html
    #   but let's do a direct newey_west
    #   For simplicity, use statsmodels' "acovf" with adjusted indexing:
    autocov = sm.tsa.stattools.acovf(d - d_bar, fft=False)
    # Bandwidth ~ h-1 for h-step ahead forecast
    # But some references choose a different bandwidth. This is a common approach:
    max_lag = h - 1 if (h - 1) >= 1 else 0
    
    gamma0 = autocov[0]
    if max_lag > 0 and len(autocov) > max_lag:
        gamma = 2 * np.sum(autocov[1:max_lag+1])
    else:
        gamma = 0.0
    
    var_d = (gamma0 + gamma) / T
    
    # 6) DM statistic
    DM_stat = d_bar / np.sqrt(var_d)

    # 7) Under H0, DM_stat ~ approx N(0,1)
    #   Two-sided p-value:
    p_value = 2 * (1 - sm.distributions.norm.cdf(np.abs(DM_stat)))

    return DM_stat, p_value

# -----------------------------------------------------------------
# 3. Modified Diebold-Mariano (M-DM) Test
# -----------------------------------------------------------------
def modified_dm_test(y_true, y_pred1, y_pred2, h=1, loss_function='SE'):
    """
    Modified DM test for small-sample bias correction.
    (Harvey, Leybourne, and Newbold, 1997).
    :return: (M-DM statistic, p-value)
    """
    # First do standard DM
    DM_stat, p_value = diebold_mariano_test(y_true, y_pred1, y_pred2, h, loss_function)

    T = len(y_true)
    # Correction factor c_T
    # Some references use: c_T = ((T + 1 - 2*h + h*(h-1)/T) / T)**0.5
    # We'll do the typical approach from HLN(1997):
    c_T = np.sqrt((T + 1 - 2*h + (h*(h-1))/T) / T)
    
    MDM_stat = DM_stat * c_T
    # Recompute p-value with normal approximation
    MDM_p_value = 2 * (1 - sm.distributions.norm.cdf(np.abs(MDM_stat)))

    return MDM_stat, MDM_p_value

# -----------------------------------------------------------------
# 4. Model Confidence Set (MCS)
# -----------------------------------------------------------------
def model_confidence_set_test(models_predictions, y_true, alpha=0.05, loss_function='SE'):
    """
    A simplified approach to the Model Confidence Set (MCS) procedure.
    For each model, we compute average loss, compare pairs using
    DM or MDM to iteratively remove poor performers.

    :param models_predictions: dict { 'model_name': np.array_of_preds, ... }
    :param y_true: actual values (array-like)
    :param alpha: significance level
    :param loss_function: 'SE' or 'AE'
    :return: A set/list of model names that remain in the MCS.
    """
    # 1) Build an initial set S of all models
    model_names = list(models_predictions.keys())
    S = set(model_names)
    
    # 2) Repeatedly test whether any model in S can be removed
    # until no more removals or only one left
    removed = True
    while removed and len(S) > 1:
        removed = False
        # For each pair (i, j) in S, do a DM or M-DM test
        # If one model is significantly worse, remove it
        to_remove = []
        model_list = list(S)
        
        for i in range(len(model_list)):
            for j in range(i+1, len(model_list)):
                m1 = model_list[i]
                m2 = model_list[j]
                
                preds1 = models_predictions[m1]
                preds2 = models_predictions[m2]
                
                # Do standard DM or MDM. We'll do DM for simplicity:
                DM_stat, p_value = diebold_mariano_test(
                    y_true, preds1, preds2, h=1, loss_function=loss_function
                )
                
                # If p-value < alpha => we can say one model is better
                # We can check the sign of DM_stat to see which model is better
                if p_value < alpha:
                    if DM_stat > 0:
                        # Model1 is significantly worse than Model2
                        to_remove.append(m1)
                    else:
                        # Model2 is significantly worse than Model1
                        to_remove.append(m2)
        
        if to_remove:
            removed = True
            # Remove from S
            for m in to_remove:
                if m in S:
                    S.remove(m)

    # S now should be the final MCS
    return list(S)
