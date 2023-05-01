import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
import scipy.stats as stats
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from optparse import OptionParser

usage="usage: %prog [options]"
parser = OptionParser(usage=usage)
parser.add_option("-f", "--file",  dest="file", help="immuno matrix [REQUIRED]", type="string")
parser.add_option("-k", "--numGridSearch",  dest="gridCV", default=100, help="Number of kfold cross-validation for GridSearch to perform [REQUIRED]", type="int")
parser.add_option("-a", "--scoremetric",  dest="metric", default='roc_auc', help="scoring method to use [REQUIRED]", type="string")
parser.add_option("-o", "--output", dest="outdir", default=os.getcwd(), help="Output directory [REQUIRED]", type="string")
parser.add_option("-s", "--save", dest="save", help="file prefixe [REQUIRED]", type="string")
# parser.add_option("-g", "--groups", dest="groups", default='Reshuffle_groups.txt', help="group_file", type="string")

options, args = parser.parse_args()


def gridSearch(X, Y, repeat, n_splits, scorer, mod, hyperparameters,  n_jobs=None, stratify=True):
    # GridSearch Wrapper Fucntion
    print("Number of repeats run is: " + str(repeat))
    dfL = []
    for i in range(0,repeat):
        if stratify==True:
            cv = StratifiedKFold(n_splits=n_splits, random_state=i, shuffle=True)
        else:
            cv = KFold(n_splits=n_splits, random_state=i, shuffle=True)
        boosted_grid = GridSearchCV(mod, hyperparameters, scoring=scorer, cv=cv, verbose=0, refit=True, error_score=np.nan, return_train_score=True, n_jobs=n_jobs) #n_jobs=n_jobs,
        grid_fit = boosted_grid.fit(X, Y)
        DF = pd.DataFrame(grid_fit.cv_results_)
        DF['Iteration'] = i
        dfL.append(DF)
    DFall = pd.concat(dfL)
    return DFall


def OverSampler(parentDIR,df, xfilename, yfilename):
    # Oversample small class to balanced data
    os.chdir(parentDIR)
    df = df.dropna()
    X = df.iloc[:,4:].copy()
    X['TimeGroup'] = df.TimeGroup.copy()
    X = pd.get_dummies(X)
    Y = df.Status.copy()

    ros = RandomOverSampler(random_state=0)
    Xoversampled, Yoversampled = ros.fit_resample(X, Y)

    if not os.path.isdir('OverSampled'):
        os.makedirs('OverSampled')
    os.chdir('OverSampled')

    Xoversampled.to_csv(xfilename, sep='\t')
    Yoversampled.to_csv(yfilename, sep='\t')
    return Xoversampled, Yoversampled

df = pd.read_csv(options.file, sep='\t', index_col=0)
df.dropna(inplace=True)
Y = df.Status.copy()

## dropping columns 'index', 'PatientID', 'SerumID', 'TimeGroup', 'Status', 'Type', 'Rep'
## Should use the below instead of iloc for increased transparency.
# X=df.drop(['index', 'PatientID', 'SerumID', 'TimeGroup', 'Status', 'Type', 'Rep'], axis=1)
X = df.iloc[:,7:].copy()

Y.replace('Melioid', 1, inplace=True)
Y.replace('Negative', 0, inplace=True)

if not os.path.isdir(options.outdir):
    os.makedirs(options.outdir)
os.chdir(options.outdir)
RAND=np.random.RandomState(4)

param_space = {'C':np.logspace(-5,2,6),
                'l1_ratio':[float(x) for x in np.linspace(0.1,0.9,7)]}
enet = LogisticRegression(penalty = 'elasticnet', solver = 'saga', max_iter=int(1e6))

Enet = gridSearch(X=X, Y=Y, repeat=options.gridCV, n_splits=10, scorer=options.metric, mod=enet, hyperparameters=param_space, n_jobs=-1, stratify=True) #, n_jobs=numCores
Enet.to_csv(options.save+'_EnetBurkPx_GridSearch.txt', sep='\t')
Enet['params2'] = Enet['params'].astype(str)
print('Finished Enet')


param_space = {'C':np.logspace(-5,2,10),
                'penalty':['l1', 'l2']}
lr = LogisticRegression( solver = 'saga', max_iter=int(1e6))

LR = gridSearch(X=X, Y=Y, repeat=options.gridCV, n_splits=10, scorer=options.metric, mod=lr, hyperparameters=param_space, n_jobs=-1, stratify=True) #, n_jobs=numCores
LR.to_csv(options.save+'_LRBurkPx_GridSearch.txt', sep='\t')
LR['params2'] = LR['params'].astype(str)
print('Finished LR')

param_space = {'learning_rate': np.logspace(-3, -1,4),
                'n_estimators': [10, 25, 50],
                'subsample': np.linspace(0.1,0.7,3),
                'colsample_bytree':np.linspace(0.05, 0.25, 3),
                'max_depth':[int(x) for x in np.linspace(3,20,3)]}

xgb =  XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=RAND) #, scale_pos_weight=len(Y1==0)/len(Y1==1) BinaryFocalLoss

LR = gridSearch(X=X, Y=Y, repeat=options.gridCV, n_splits=10, scorer=options.metric, mod=xgb, hyperparameters=param_space, n_jobs=-1, stratify=True) #, n_jobs=numCores
LR.to_csv(options.save+'_XGBBurkPx_GridSearch.txt', sep='\t')
LR['params2'] = LR['params'].astype(str)


###### FIGS
# if not os.path.isdir('Figures'):
#     os.makedirs('Figures')
# os.chdir('Figures')

# LR = pd.read_csv('W1_EnetBurkPx_GridSearch.txt', sep='\t', index_col=0)
# LR['params2'] = LR['params'].astype(str)
# fig, ax = plt.subplots()
# sns.boxplot(data=LR, y= 'mean_test_score', x='params2', ax=ax)
# plt.xticks(rotation=90)
# fig.savefig('W1_Enet.png', dpi=300, bbox_inches="tight", transparent=False)


# LR = pd.read_csv('W1_LRBurkPx_GridSearch.txt', sep='\t', index_col=0)
# LR['params2'] = LR['params'].astype(str)
# fig, ax = plt.subplots()
# sns.boxplot(data=LR, y= 'mean_test_score', x='params2', ax=ax)
# plt.xticks(rotation=90)
# fig.savefig('W1_LR.png', dpi=300, bbox_inches="tight", transparent=False)


# LR = pd.read_csv('W1_XGBBurkPx_GridSearch.txt', sep='\t', index_col=0)
# LR['params2'] = LR['params'].astype(str)
# fig, ax = plt.subplots()
# sns.boxplot(data=LR, y= 'mean_test_score', x='params2', ax=ax)
# plt.xticks(rotation=90)
# fig.savefig('W1_XGB.png', dpi=300, bbox_inches="tight", transparent=False)
