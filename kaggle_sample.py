import os

import catboost as cb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns

# from category_encoders import MEstimateEncoder
from optuna.integration import CatBoostPruningCallback
from sklearn.metrics import confusion_matrix, log_loss, mean_squared_error
from sklearn.model_selection import StratifiedKFold

colours = ['#217CA3', '#C4274C', '#FFC933']

# set plot theme
sns.set_theme(
    style='ticks',
    palette=colours,
    font_scale=1.25,
    rc={'figure.figsize':(12,8),
        'axes.titlesize':20,
        'axes.spines.top':False,
        'axes.spines.right':False}
)

# create empty list
dfs = []

# for loop to import train/test data
# loop through files in directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    # loop through filenames
    for filename in filenames:
        # exclude sample submission file
        if filename != 'sample_solution.csv':
            # import csv as df
            df = pd.read_csv(os.path.join(dirname, filename))
            # create column to distinguish between train/test sets
            df['dataset'] = filename[:filename.rfind('.')]
            # convert 'train' to 'val' in 20% of rows
            if df['dataset'][0] == 'train':
                df.loc[df.sample(frac=0.2).index, 'dataset'] = 'holdout'  
            # append to list
            dfs.append(df)

# combine train/test datasets 
df = pd.concat(dfs, ignore_index=True)

# check the 'dataset' column contains correct split of train/holdout/test
df['dataset'].value_counts()

# check target class distribution across train/holdout
df.groupby(['dataset'])['is_strike'].value_counts(normalize=True)

# create 'train' to keep EDA code simple
train = df.loc[df.dataset=='train']

train.nunique()

train.head()

# target label distribution
train['is_strike'].value_counts(normalize=True)

# distribution of target (is_strike)
sns.countplot(data=train, x='is_strike')

plt.title('Distribution of Strikes (0 = Not Strike, 1 = Strike)')
plt.tight_layout()
plt.show()

# drop uid, is_strike, and dataset for plot dataframe
plt_df = df.drop(['uid', 'is_strike'], axis=1)

# convert boolean columns to int
for col in plt_df.columns:
    if plt_df[col].nunique() <= 15:
        plt_df[col] = plt_df[col].astype('category')

# specify columns and rows
ncols = 3
nrows = int(np.ceil(len(plt_df.select_dtypes(include=np.number).columns)/ncols))

# create figure and axes
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*8,nrows*4))

for i, col in enumerate(plt_df.select_dtypes(include=np.number).columns):
    ax = axes[i//ncols, i%ncols]
    sns.histplot(x=plt_df.loc[plt_df.dataset=='train'][col], ax=ax)
    sns.histplot(x=plt_df.loc[plt_df.dataset=='holdout'][col], ax=ax)
    sns.histplot(x=plt_df.loc[plt_df.dataset=='test'][col], ax=ax)
    ax.set_ylabel('')

plt.suptitle('Numerical Feature Distributions', fontsize=20, y=1)
plt.tight_layout()
plt.show()

# specify columns and rows
ncols = 3
nrows = int(np.ceil(len(plt_df.drop('dataset', axis=1).select_dtypes(exclude=np.number).columns)/ncols))

# create figure and axes
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*8,nrows*5))

for i, col in enumerate(plt_df.drop('dataset', axis=1).select_dtypes(exclude=np.number).columns):
    ax = axes[i//ncols, i%ncols]
    sns.countplot(x=plt_df.loc[plt_df.dataset=='train'][col], ax=ax, color='#217CA3')
    sns.countplot(x=plt_df.loc[plt_df.dataset=='holdout'][col], ax=ax, color='#C4274C')
    sns.countplot(x=plt_df.loc[plt_df.dataset=='test'][col], ax=ax, color='#FFC933')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_ylabel('')

plt.suptitle('Categorical Feature Distributions', fontsize=20, y=1)
plt.tight_layout()
plt.show()

print(f'Outfield Fielding Alignment\n')

print('Train Counts:')
print(f"{df.loc[df.dataset == 'train'].of_fielding_alignment.value_counts()}\n")

print('Holdout Counts:')
print(f"{df.loc[df.dataset == 'holdout'].of_fielding_alignment.value_counts()}\n")

print('Test Counts:')
print(f"{df.loc[df.dataset == 'test'].of_fielding_alignment.value_counts()}\n")

print(f'Balls\n')

print('Train Counts:')
print(f"{df.loc[df.dataset == 'train'].balls.value_counts()}\n")

print('Holdout Counts:')
print(f"{df.loc[df.dataset == 'holdout'].balls.value_counts()}\n")

print('Test Counts:')
print(f"{df.loc[df.dataset == 'test'].balls.value_counts()}\n")

print(f'Pitch Type\n')

print('Train Counts:')
print(f"{df.loc[df.dataset == 'train'].pitch_type.value_counts()}\n")

print('Holdout Counts:')
print(f"{df.loc[df.dataset == 'holdout'].pitch_type.value_counts()}\n")

print('Test Counts:')
print(f"{df.loc[df.dataset == 'test'].pitch_type.value_counts()}\n")

# horizontal and vertical plate
sns.scatterplot(data=train, x='plate_x', y='plate_z', hue='is_strike')

plt.title('Strikes Plotted Over Horizontal and Vertical Plate')
plt.tight_layout()
plt.show()