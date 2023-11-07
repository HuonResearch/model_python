import pandas as pd
import statsmodels.api as sm
import pymc3 as pm
from statsmodels.miscmodels.ordinal_model import OrderedModel

# Read data
data = pd.read_csv("raw_data/responses_with_human_coding.csv")

# Data manipulation
data['Prompt_n'] = pd.Categorical(data['Prompt_n'], categories=["Name", "Describe", "Simulate", "Example"])
data['Temperature'] = pd.Categorical(data['Temperature'])
data['Role_n'] = pd.Categorical(data['Role_n'], categories=["Helpful", "Expert"])
data['Shot_n'] = pd.Categorical(data['Shot_n'], categories=["Zero", "One", "Few"])
data['Version'] = pd.Categorical(data['Version'])

data['consistency'] = data[['consistency_coder_1', 'consistency_coder_2']].mean(axis=1, skipna=True).round()
data['decency'] = data[['decency_coder_1', 'decency_coder_2']].mean(axis=1, skipna=True).round()

# Model data using statsmodels for ordinal logistic regression
mod_consistency = OrderedModel(data['consistency'],
                               data[['Version', 'Prompt_n', 'Temperature', 'Role_n', 'Shot_n']],
                               distr='logit')
res_consistency = mod_consistency.fit(method='bfgs')

mod_decency = OrderedModel(data['decency'],
                           data[['Version', 'Prompt_n', 'Temperature', 'Role_n', 'Shot_n']],
                           distr='logit')
res_decency = mod_decency.fit(method='bfgs')


