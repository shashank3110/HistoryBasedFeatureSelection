#%%
# HBFS : https://github.com/Brett-Kennedy/HistoryBasedFeatureSelection
# MRMR : https://pypi.org/project/mrmr-selection/

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.datasets import fetch_openml
from sklearn.metrics import f1_score
from history_based_feature_selection import test_all_features, feature_selection_history
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
#%%
# Collect the data to be used.

n_features = 30
X, Y = make_classification(n_samples=20000, n_features=n_features, n_informative = 8, 
                           n_classes = 4, weights=[0.2,0.6,0.15,0.05]) # creating imbalanced class dataset
col_names = [f'col_{i+1}' for i in range(n_features)]

X = pd.DataFrame(X, columns=col_names)
Y = pd.DataFrame(Y, columns=['target'])

# Oversampling for imbalanced classes
sm = SMOTE(sampling_strategy='auto', random_state=42)
X, Y = sm.fit_resample(X,Y)
# train test split
x, x_test, y, y_test = train_test_split(X,Y, test_size=0.3)

# Divide the train data into train and validate sets
x_train, x_val, y_train, y_val = train_test_split(x,y, test_size=0.3)

#%%
# Execute feature_selection_history(). This returns a dataframe tha lists the feature
# sets that were tested and their scores on the validation set. 

model = DecisionTreeClassifier()
scores_df = feature_selection_history(
        model, {},
        x_train, y_train, x_val, y_val,
        num_iterations=10, num_estimates_per_iteration=5_000, num_trials_per_iteration=25, 
        max_features=None, penalty=None, higher_is_better=True,
        plot_evaluation=True, verbose=True, draw_plots=True,
        metric=f1_score, metric_args={'average':'macro'})
#%%
scores_df.columns = [f'col_{i+1}' for i in range(n_features)] + scores_df.columns[-2:].tolist()
#%%
selected_cols = [ col for col,val in scores_df.iloc[0,:-2].items() if val=='Y' ]

#%%
# scores_df.to_csv('results.csv')
scores_df.to_csv('results_with_oversampling_v2.csv')
print(f'Final output: {scores_df}')
# %%
# baseline model i.e.  with all features
baseline_model = RandomForestClassifier(random_state=42) #DecisionTreeClassifier(random_state=42)
baseline_model.fit(x_train, y_train)

y_val_base_pred = baseline_model.predict(x_val)

print(classification_report(y_val, y_val_base_pred))

# %%
#  model with selected features

model = RandomForestClassifier(random_state=42) #DecisionTreeClassifier(random_state=42)
x_train_sel = x_train[selected_cols]
x_val_sel = x_val[selected_cols]
model.fit(x_train_sel, y_train)

y_val_pred = model.predict(x_val_sel)

print(classification_report(y_val, y_val_pred))

# %%
from mrmr import mrmr_classif 
mrmr_sel_features = mrmr_classif(x_train, y_train, K = 8 )

# %%
model2 = RandomForestClassifier(random_state=42) #DecisionTreeClassifier(random_state=42)
x_train_mrmr_sel = x_train[mrmr_sel_features]
x_val_mrmr_sel = x_val[mrmr_sel_features]
model2.fit(x_train_mrmr_sel, y_train)

y_val_mrmr_pred = model2.predict(x_val_mrmr_sel)

print(classification_report(y_val, y_val_mrmr_pred))
# %%
