import numpy as np
import pandas as pd
from scipy.stats import norm

#50/50 
N_treatment = 10_000
N_control = 10_000

# Generating clicks distribution
click_treatment = pd.Series(np.random.binomial(1,0.4,N_treatment))
click_control = pd.Series(np.random.binomial(1,0.2,N_control))

# Generating ids
treatment_id = pd.Series(np.repeat('treatment', N_treatment))
control_id = pd.Series(np.repeat('control', N_control))

df_treatment = pd.concat([click_treatment,treatment_id], axis=1)
df_control = pd.concat([click_control,control_id], axis=1)

df_treatment.columns = ['click', 'group']
df_control.columns = ['click', 'group']

# Creating the AB test dataset
df_abtest = pd.concat([df_treatment, df_control], axis=0).reset_index(drop=True)


x_treatment = df_abtest.groupby('group')['click'].sum().loc['treatment']
x_control = df_abtest.groupby('group')['click'].sum().loc['control']

print('Number of clicks in treatment:', x_treatment)
print('Number of clicks in control:', x_control)

prob_treatment_hat = x_treatment/N_treatment
prob_control_hat = x_control/N_control

print('Click probability in Treatment group:', prob_treatment_hat)
print('Click probability in Control group:', prob_control_hat)

p_pooled_hat = (x_treatment+x_control)/(N_treatment+N_control)
pooled_variance = p_pooled_hat * (1-p_pooled_hat) * (1/N_treatment + 1/N_control)
std_error = np.sqrt(pooled_variance)

print('p^pooled is:', p_pooled_hat)
print('pooled_variance is:', pooled_variance)
print('standard error is:', std_error)


test_statistics = (prob_control_hat - prob_treatment_hat)/std_error
print('Test statistics for 2-sample Z test is:', test_statistics)

alpha = 0.05
print('Alpha: significance level is:', alpha)

Z_criteria = norm.ppf(1-alpha/2)
print('Z critical value from standard normal distribution:', Z_criteria)

p_value = 2 * norm.sf(abs(test_statistics))
print('p-value of the 2-sample Z-test is:', round(p_value,3))

# confidence interval
ci = [round((prob_treatment_hat - prob_control_hat) - std_error*Z_criteria, 3),
      round((prob_treatment_hat - prob_control_hat) + std_error*Z_criteria, 3)]
print('Confidence Interval of the 2 sample Z-test is:', ci)

# minimum detectable effect
delta = 0.03
