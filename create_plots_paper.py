from itertools import combinations
from random import choices
from random import seed
from random import randint

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import paired_manhattan_distances
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from lir import KDECalibrator, ELUBbounder, CalibratedScorer, metrics, plotting, LogitCalibrator
from lir.transformers import AbsDiffTransformer

csv_url = 'https://raw.githubusercontent.com/NetherlandsForensicInstitute/elemental_composition_glass/main/duplo.csv'
data_set = pd.read_csv(csv_url, delimiter=',').rename(columns={'Item': 'Subject', 'Piece': 'Repeat', 'id': 'Id'})

# Make sure that the sorting is first by item, then by repeat
data_set = data_set.sort_values(by=["Subject", "Repeat"], axis=0, ascending=True)

# Only keep the first 400 rows, as the free version of google colab
# does not have enough RAM available for computations on the full set
data_set = data_set.drop(data_set.index[400:])

variables = ["K39", "Ti49", "Mn55", "Rb85", "Sr88", "Zr90", "Ba137", "La139", "Ce140", "Pb208"]
labels = ["Subject"]

obs = data_set[variables].to_numpy()
ids = data_set[labels].to_numpy()

# First we split off 20% from the data for a hold-out validation set (grouped per glass particle)
splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state=1)
split = splitter.split(obs, groups=ids)
train_select_indices, val_indices = next(split)

# Then we split off 20% to use as a test set
splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state=1)
split = splitter.split(obs[train_select_indices], groups=ids[train_select_indices])
train_indices, select_indices = next(split)

# We create the train, selection and validation sets
# obs are the concentrations, ids are the corresponding labels indicating the source item
ids_train = ids[train_indices]
ids_select = ids[select_indices]
ids_val = ids[val_indices]
obs_train = obs[train_indices]

# show the sizes of the data sets
print(f'Size of total data set: {len(obs)}')

print(f'Size of training set: {len(ids_train)}')
print(f'Size of selection set: {len(ids_select)}')
print(f'Size of validation set: {len(ids_val)}')

z_score_transformer = StandardScaler()
z_score_transformer.fit(obs_train)
obs_zscore = z_score_transformer.transform(obs)

plt.hist(obs[:,0], bins=20, histtype='step', color='tab:purple', alpha=1, label='before pre-processing')
plt.hist(obs_zscore[:,0], bins=20, color='tab:cyan', alpha=0.5, label='after pre-processing')
plt.xlabel('K39 value')
plt.ylabel('count')
plt.legend()
plt.savefig('fig4_step3preprocess_both.jpeg', edgecolor='white', dpi=600, facecolor='white', transparent=False)


def create_pairs(obs, ids):
    """
    creates all possible pairs between the items represented by ids
    the ids refer to the total set of observations given by obs.
    """
    # Create matrices with item IDs that refer to the total data set
    H1_ids = np.transpose(np.tile(np.unique(ids), (2, 1)))
    H2_ids = np.asarray(list(combinations(np.unique(ids), 2)))

    # For H1-data: use the first repeat for each item in the first colum,
    # and the second repeat of that item in the second.
    # It is assumed: that obs is sorted first by item ID, then by repeat ID;
    # that all items have exactly 2 repeats; that there are no missing items.
    H1_obs_rep_1 = obs[2*H1_ids[:,0] - 2]
    H1_obs_rep_2 = obs[2*H1_ids[:,1] - 1]
    H1_obs_pairs = np.stack((H1_obs_rep_1, H1_obs_rep_2), axis=2)

    # For H2-data: use for both items their first repeats
    H2_obs_item_1 = obs[2*H2_ids[:,0] - 2]
    H2_obs_item_2 = obs[2*H2_ids[:,1] - 2]
    H2_obs_pairs = np.stack((H2_obs_item_1, H2_obs_item_2), axis=2)

    # Combine the H1 and H2 data, and create vector with classes: H1 and H2
    obs_pairs = np.concatenate((H1_obs_pairs, H2_obs_pairs))
    hypothesis = np.concatenate((np.array(['H1']*len(H1_ids)),
                            np.array(['H2']*len(H2_ids))))

    return obs_pairs, hypothesis


obs_pairs_train, hypothesis_train = create_pairs(obs_zscore, ids_train)
obs_pairs_select, hypothesis_select = create_pairs(obs_zscore, ids_select)

dissimilarity_scores_train = paired_manhattan_distances(obs_pairs_train[:,:,0], obs_pairs_train[:,:,1])
dissimilarity_scores_select = paired_manhattan_distances(obs_pairs_select[:,:,0], obs_pairs_select[:,:,1])

with plotting.show() as ax:
    ax.score_distribution(scores=dissimilarity_scores_train, y=(hypothesis_train=='H1')*1, bins=np.linspace(0, 30, 60), weighted=True)
    plt.xlabel('dissimilarity score')
    H1_legend = mpatches.Patch(color='tab:blue', alpha=.5, label='$H_1$-true')
    H2_legend = mpatches.Patch(color='tab:orange', alpha=.5, label='$H_2$-true')
    ax.legend(handles=[H1_legend, H2_legend])
    ax.savefig('fig5A_step4dissimilarity_scores.jpeg', edgecolor='white', dpi=600, facecolor='white', transparent=False)


# machine learning models need a single vector as input. The AbsDiffTransformer takes two feature vectors,
# one for each subject of a pair, and returns the elementwise absolute differences
# The AbsDiffTransformer and support vector machine (SVC) are combined into a single pipeline using sklearns Pipeline class.
machine_learning_scorer = Pipeline([('abs_difference', AbsDiffTransformer()), ('classifier', SVC(probability=True))])

# the model has to be fit on the data
machine_learning_scorer.fit(obs_pairs_train, hypothesis_train=="H1")
# score can be computed using the 'predict_proba' function. This is another sklearn convention,
# which returns two columns of which we take the second using '[:,1]'
machine_learning_scores_train = machine_learning_scorer.predict_proba(obs_pairs_train)[:, 1]

with plotting.show() as ax:
    ax.score_distribution(scores=machine_learning_scores_train, y=(hypothesis_train=='H1')*1, bins=np.linspace(0, 1, 10), weighted=True)
    plt.xlabel('machine learning score')
    H1_legend = mpatches.Patch(color='tab:blue', alpha=.5, label='$H_1$-true')
    H2_legend = mpatches.Patch(color='tab:orange', alpha=.5, label='$H_2$-true')
    ax.legend(handles=[H1_legend, H2_legend])
    ax.savefig('fig5B_step4ML_scores.jpeg', edgecolor='white', dpi=600, facecolor='white', transparent=False)

kde_calibrator = KDECalibrator(bandwidth='silverman')

kde_calibrator.fit(dissimilarity_scores_train, hypothesis_train=="H1")
with plotting.show() as ax:
    ax.calibrator_fit(kde_calibrator, score_range=[0, 30])
    ax.score_distribution(scores=dissimilarity_scores_train, y=(hypothesis_train=='H1')*1, bins=np.linspace(0, 30, 60), weighted=True)
    ax.xlabel('dissimilarity score')
    H1_legend = mpatches.Patch(color='tab:blue', alpha=.5, label='$H_1$-true')
    H2_legend = mpatches.Patch(color='tab:orange', alpha=.5, label='$H_2$-true')
    ax.legend(handles=[H1_legend, H2_legend])
    ax.savefig('fig6_step5kde.jpeg', edgecolor='white', dpi=600, facecolor='white', transparent=False)

logreg_calibrator = LogitCalibrator()

bounded_kde_calibrator = ELUBbounder(kde_calibrator)
bounded_logreg_calibrator = ELUBbounder(logreg_calibrator)

lrs_train_logreg = bounded_logreg_calibrator.fit_transform(dissimilarity_scores_train, hypothesis_train=="H1")
lrs_train_kde = bounded_kde_calibrator.fit_transform(dissimilarity_scores_train, hypothesis_train=="H1")

plt.scatter(dissimilarity_scores_train, np.log10(lrs_train_logreg), color='tab:purple', marker=".", label='logistic regression')
plt.scatter(dissimilarity_scores_train, np.log10(lrs_train_kde), color='tab:cyan', label='KDE')
plt.xlim([0, 5])
plt.legend()
plt.xlabel('dissimilarity score')
plt.ylabel('log$_{10}$(LR)')
plt.savefig('fig7_step5kde_logreg.jpeg', edgecolor='white', dpi=600, facecolor='white', transparent=False)

print(f'ELUB log(LR) bounds for logreg are {np.log10(bounded_logreg_calibrator._lower_lr_bound):.2f} and {np.log10(bounded_logreg_calibrator._upper_lr_bound):.2f}')
print(f'ELUB log(LR) bounds for kde are {np.log10(bounded_kde_calibrator._lower_lr_bound):.2f} and {np.log10(bounded_kde_calibrator._upper_lr_bound):.2f}')


def show_performance(lrs, hypothesis, calibrator, fileprefix):
      hypothesis=(hypothesis=='H1')*1 # convert to 0 or 1 for technical reasons
      # show the distribution of LRs together with the ELUB values
      print('Histogram of the LRs:\n')

      with plotting.show() as ax:
          ax.lr_histogram(lrs, hypothesis)
          H1_legend = mpatches.Patch(color='tab:blue', alpha=.5, label='$H_1$-true')
          H2_legend = mpatches.Patch(color='tab:orange', alpha=.5, label='$H_2$-true')
          ax.legend(handles=[H1_legend, H2_legend])
          plt.savefig(f'{fileprefix}_hist.jpeg', edgecolor='white', dpi=600, facecolor='white', transparent=False);


      print(f'\n ELUB log LR bounds are {np.log10(calibrator._lower_lr_bound):.2f} and {np.log10(calibrator._upper_lr_bound):.2f} \n')

      print('PAV plot (closer to the line y=x is better):\n')
      # show the PAV plot (closer to the line y=x is better)
      with plotting.show() as ax:
          ax.pav(lrs, hypothesis)
          plt.savefig(f'{fileprefix}_pav.jpeg', edgecolor='white', dpi=600, facecolor='white', transparent=False)

      with plotting.show() as ax:
          ax.pav_zoom(lrs, hypothesis)
          plt.savefig(f'{fileprefix}_pav_zoom.jpeg', edgecolor='white', dpi=600, facecolor='white', transparent=False)

      # print the quality of the system as log likelihood ratio cost (lower is better)
      print(f'\n The log likelihood ratio cost is {metrics.cllr(lrs, hypothesis):.3f} (lower is better)\n')


# Selected system: support vector machine + KDE

# The whole LR system consists of computing the score and then mapping the scores to lrs
# we use the machine learning scorer introduced in step 4
scorer = Pipeline([('abs_difference', AbsDiffTransformer()), ('classifier', SVC(probability=True))])

lr_system = CalibratedScorer(scorer, bounded_kde_calibrator)

# we fit the whole system. When fitting (=training) a CalibratedScorer, both the machine learning model and the transformation to LRs are trained on the supplied data
lr_system.fit(obs_pairs_train, hypothesis_train == "H1")
lrs_select = lr_system.predict_lr(obs_pairs_select)

show_performance(lrs_select, hypothesis_select, lr_system.calibrator, 'fig8_step6_selection')

# create the combined data set
obs_train_select = obs[train_select_indices]
ids_train_select = ids[train_select_indices]

# step 3 pre-processing: normalise
z_score_transformer.fit(obs_train_select)
obs_zscore = z_score_transformer.transform(obs)

# step 4: combine the pairs into one feature vector by taking the absolute difference
obs_pairs_train_select, hypothesis_train_select = create_pairs(obs_zscore, ids_train_select)
obs_pairs_val, hypothesis_val = create_pairs(obs_zscore, ids_val)

selected_lr_system = CalibratedScorer(machine_learning_scorer, ELUBbounder(KDECalibrator(bandwidth='silverman')))
# step 4+5 combined: we fit the whole system
selected_lr_system.fit(obs_pairs_train_select, hypothesis_train_select == 'H1')

# compute the LRs on the validation data
lrs_val = selected_lr_system.predict_lr(obs_pairs_val)

# we always inspect the characteristics we also look at in selection
show_performance(lrs_val, hypothesis_val, selected_lr_system.calibrator, 'fig9_step7_validate')

# There are many other characteristics that we may want to inspect, such as the empirical cross entropy (ECE) plot.
with plotting.show() as ax:
      ax.ece(lrs_val, hypothesis_val == 'H1')
      plt.savefig(f'fig9_step7_ece.jpeg', edgecolor='white', dpi=600, facecolor='white', transparent=False)

with plotting.show() as ax:
    ax.ece_zoom(lrs_val, hypothesis_val == 'H1')
    plt.savefig(f'fig9_step7_ece_zoom.jpeg', edgecolor='white', dpi=600, facecolor='white', transparent=False)


# create casework LR system
# create new pairs for the combined data
obs_zscore = z_score_transformer.fit_transform(obs)
obs_pairs, hypothesis = create_pairs(obs_zscore, ids)

# fit the same system on all the data
selected_lr_system.fit(obs_pairs, hypothesis=='H1')

observation_1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
observation_2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

manual_entered_data = np.moveaxis(np.array([z_score_transformer.transform([observation_1, observation_2])]), 1, -1)

print(f'The LR obtained is {selected_lr_system.predict_lr(manual_entered_data)[0]}')



# Plaatje overfitting

from scipy.interpolate import UnivariateSpline

np.random.seed(1)

#x and y array definition (initial set of data points)
x = np.linspace(0, 10, 30)
# y = np.sin(0.5*x)*np.sin(x*np.random.randn(30))
y = 1/2*x +  np.random.normal(0, 1, size=len(x))


def func(x):
    return 1/2*x


#spline definition
spline = UnivariateSpline(x, y, s=5)

x_spline = np.linspace(0, 10, 1000)
y_spline = spline(x_spline)

y_line = func(x_spline)

# Changing the smoothing factor for a better fit
spline.set_smoothing_factor(0.05)
y_spline2 = spline(x_spline)

# Plotting
fig = plt.figure()
ax = fig.subplots()
ax.scatter(x, y, label='observations')
ax.plot(x_spline, y_line, linestyle='dashed', label='true model')
ax.plot(x_spline, y_spline2, color='tab:orange', label='overfitted model')
plt.legend()
plt.savefig('fig2_overfitting.jpeg', edgecolor='white', dpi=600, facecolor='white', transparent=False)
