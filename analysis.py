import numpy as np
import pandas as pd

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

df_abtest = pd.concat([df_treatment, df_control], axis=0).reset_index(drop=True)
