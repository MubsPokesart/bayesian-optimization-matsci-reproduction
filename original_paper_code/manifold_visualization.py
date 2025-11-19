import numpy as np
import torch
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import GPyOpt
import GPy
import random
import os
import matplotlib as mpl
import matplotlib.tri as tri
import ternary
import pickle
import datetime
from collections import Counter
import matplotlib.ticker as ticker
import pyDOE
import random
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import decomposition
import plotly
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.font_manager as font_manager
import copy

# random state for PCA
rs = 0

# Load dataset
# dataset names = ['Crossed barrel', 'Perovskite', 'AgNP', 'P3HT', 'AutoAM']
dataset_name = 'Crossed barrel'
raw_dataset = pd.read_csv(dataset_name + '_dataset.csv')
feature_name = list(raw_dataset.columns)[:-1]
objective_name = list(raw_dataset.columns)[-1]

# Formulate optimization as global minimization
ds = copy.deepcopy(raw_dataset)
# only P3HT/CNT, Crossed barrel, AutoAM need this line; Perovskite and AgNP do not need this line.
ds[objective_name] = -raw_dataset[objective_name].values

# Process dataset for pool-based active learning
ds_grouped = ds.groupby(feature_name)[objective_name].agg(lambda x: x.unique().mean())
ds_grouped = (ds_grouped.to_frame()).reset_index()

# Normalize input features and objective values
s_scaler = preprocessing.StandardScaler()
ds_normalized_values = s_scaler.fit_transform(ds_grouped[list(raw_dataset.columns)].values)
ds_normalized = pd.DataFrame(ds_normalized_values, columns=list(raw_dataset.columns))

# Histogram of objective values
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.histplot(ds_normalized[objective_name], stat="probability", discrete=False, bins=40, element='step', color='#0570b0')
ax.set_xlim([-4.5, 4.5])
ax.set_ylim([0, 0.21])
ax.set_ylabel('Probability', fontsize=33, fontname="Arial")
ax.set_xlabel(objective_name, fontsize=33, fontname="Arial")
ax.xaxis.set_tick_params(labelsize=33)
plt.xticks([-4, -2, 0, 2, 4], ['-4', '-2', '0', '2', '4'], fontname='Arial')
plt.yticks([0, 0.05, 0.1, 0.15, 0.2], ['0', '0.05', '0.10', '0.15', '0.20'], fontname='Arial')
ax.yaxis.set_tick_params(labelsize=33)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
font = font_manager.FontProperties(family='Arial', size=33, style='normal')
leg = plt.legend(labels=[dataset_name], handlelength=0.7, handletextpad=0.4, prop=font, frameon=False, loc='upper left')

# Perform PCA
pca_ = decomposition.PCA(n_components=3, random_state=rs)
pca_.fit(ds_normalized[feature_name])
X_pca_values = pca_.transform(ds_normalized[feature_name])

print(X_pca_values.shape)
print(pca_.explained_variance_ratio_)

X_pca = pd.DataFrame()
X_pca['PC1'] = X_pca_values[:, 0]
X_pca['PC2'] = X_pca_values[:, 1]
X_pca['PC3'] = X_pca_values[:, 2]
X_pca[objective_name] = ds_normalized[objective_name]

# Visualize in 3D
fig = px.scatter_3d(X_pca, x='PC1', y='PC2', z='PC3',
                    color=objective_name, size_max=20, opacity=0.6)

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.update_layout(
    autosize=False,
    scene=dict(
        xaxis=dict(autorange=False, nticks=3, range=[-2.5, 4],
                   backgroundcolor="rgb(215, 200,215)", gridcolor="white", gridwidth=2,
                   linecolor="rgb(255, 255, 255)", linewidth=10, showbackground=True,
                   zerolinecolor="rgb(255, 255, 255)", zeroline=True, zerolinewidth=10,
                   showgrid=True, visible=True, showticklabels=False),
        yaxis=dict(autorange=False, nticks=3, range=[-2.4, 5],
                   backgroundcolor="rgb(215, 200,215)", gridcolor="white", gridwidth=2,
                   linecolor="rgb(255, 255, 255)", linewidth=10, showbackground=True,
                   zerolinecolor="rgb(255, 255, 255)", zeroline=True, zerolinewidth=10,
                   showgrid=True, visible=True, showticklabels=False),
        zaxis=dict(autorange=False, nticks=3, range=[-3, 4.5],
                   backgroundcolor="rgb(215, 200,215)", gridcolor="white", gridwidth=2,
                   linecolor="rgb(255, 255, 255)", linewidth=10, showbackground=True,
                   zerolinecolor="rgb(255, 255, 255)", zeroline=True, zerolinewidth=10,
                   showgrid=True, visible=True, showticklabels=False)
    ),
    scene_camera=dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-1.2, y=1.56, z=1.92)
    )
)

fig.update_layout(width=700, height=700, plot_bgcolor='rgb(0,0,0)')
fig.update_layout(scene_aspectmode='manual', scene_aspectratio=dict(x=1, y=1, z=1))
fig.update_layout(title=dict(text=dataset_name, x=0.5, y=0.95), font=dict(family="Arial", size=20))
fig.update_layout(font=dict(family="Arial", size=20))

fig.show()
