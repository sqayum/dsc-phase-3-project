import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
import seaborn as sns
from sklearn.model_selection import validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_confusion_matrix, roc_curve, auc, recall_score, accuracy_score, precision_score, fbeta_score, make_scorer
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data
from IPython.display import Image

def f2_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2, average='binary')

f2 = make_scorer(f2_score)

def plot_decision_tree(decision_tree, feature_labels, class_labels=['False', 'True'], *, width=None, height=None):
    dot_data = export_graphviz(decision_tree=decision_tree,
                                feature_names=feature_labels,
                                class_names=class_labels,
                                filled=True,
                                rounded=True,
                                proportion=True)

    graph = graph_from_dot_data(dot_data)
    return Image(graph.create_png(), width=width, height=height)


def get_val_metrics(y_train, y_validate, y_train_pred, y_validate_pred):
    metrics = {'Train':{}, 'Validate': {}}
    for score_name, score_func in [('Accuracy', accuracy_score), ('Recall', recall_score), ('Precision', precision_score), ('F2', f2_score)]:
        if score_name == 'Precision' or score_name == 'Recall':
            metrics['Train'].update({score_name: score_func(y_train, y_train_pred, average='binary')})
            metrics['Validate'].update({score_name: score_func(y_validate, y_validate_pred, average='binary')})
        else:
            metrics['Train'].update({score_name: score_func(y_train, y_train_pred)})
            metrics['Validate'].update({score_name: score_func(y_validate, y_validate_pred)})
    return metrics


def val_confusion_matrix(estimator, X_train, X_validate, y_train, y_validate, *, test=False):
    fig, (ax3, ax4) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

    plot_confusion_matrix(estimator, X_train, y_train, normalize="all", cmap='Blues', ax=ax3)
    ax3.grid(False)
    ax3.set(title='Confusion Matrix (Training)')

    plot_confusion_matrix(estimator, X_validate, y_validate, normalize="all", cmap='Oranges', ax=ax4)
    ax4.grid(False)
    if is_test:
        ax4.set(title='Confusion Matrix (Test)')
    else:
        ax4.set(title='Confusion Matrix (Validation)')

    return

def val_roc_curve(estimator, X_train, X_validate, y_train, y_validate):
    y_train_score = estimator.predict_proba(X_train)
    y_validate_score = estimator.predict_proba(X_validate)

    train_fpr, train_tpr, thresholds = roc_curve(y_train, y_train_score[:,1])
    validate_fpr, validate_tpr, thresholds = roc_curve(y_validate, y_validate_score[:,1])

    train_auc = round(auc(train_fpr, train_tpr), 2)
    validate_auc = round(auc(validate_fpr, validate_tpr), 2)

    return train_fpr, train_tpr, validate_fpr, validate_tpr, train_auc, validate_auc



def validate(estimator, X_train, X_validate, y_train, y_validate, *, is_test=False):
    estimator.fit(X_train, y_train)
    y_train_pred = estimator.predict(X_train)
    y_validate_pred = estimator.predict(X_validate)

    metrics = get_val_metrics(y_train, y_validate, y_train_pred, y_validate_pred)
    if is_test:
        metrics['Train'] = metrics['Validate']
        metrics.pop('Validate')

    train_fpr, train_tpr, validate_fpr, validate_tpr, train_auc, validate_auc = val_roc_curve(estimator, X_train, X_validate, y_train, y_validate)

    plt.rcParams['axes.labelsize'] = 15
    plt.rcParams['axes.titlesize'] = 18

    fig = plt.figure(figsize=(12,18))
    gs = GridSpec(3, 2, figure=fig)
    gs.update(wspace=0.3, hspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.set(title='Metrics', ylim=(0,1))
    ax1.yaxis.set_major_locator(MultipleLocator(base=0.1))
    metrics_df = pd.DataFrame(metrics).unstack().reset_index().rename(columns={'level_0': 'Dataset', 'level_1': 'Score', 0: 'Value'})
    sns.barplot(data=metrics_df, x="Score", y="Value", hue="Dataset", ax=ax1)


    ax2 = fig.add_subplot(gs[1, 0])
    plot_confusion_matrix(estimator, X_train, y_train, normalize="all", cmap='Blues', ax=ax2)
    ax2.grid(False)
    ax2.set(title='Confusion Matrix (Training)')

    ax3 = fig.add_subplot(gs[1, 1])
    plot_confusion_matrix(estimator, X_validate, y_validate, normalize="all", cmap='Oranges', ax=ax3)
    ax3.grid(False)
    if is_test:
        ax3.set(title='Confusion Matrix (Test)')
    else:
        ax3.set(title='Confusion Matrix (Validation)')

    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(train_fpr, train_tpr, color="tab:blue", label=f'Training (AUC = {train_auc})')
    if is_test:
        ax4.plot(validate_fpr, validate_tpr, color="tab:orange", label=f'Test (AUC = {validate_auc})')
    else:
        ax4.plot(validate_fpr, validate_tpr, color="tab:orange", label=f'Validation (AUC = {validate_auc})')
    ax4.plot([0,1], [0,1], color='red', ls=':')
    ax4.set(
        title='ROC Curve',
        xlabel='False Positive Rate',
        ylabel='True Positive Rate')
    ax4.legend()

    return


def plot_validation_curve(estimator, X_train, y_train, *, param_name, param_range, scoring=f2, fit_params=None, n_jobs=-1):
    estimator_name = str(estimator)

    train_scores, test_scores = validation_curve(estimator, X_train, y_train,
                                                 param_name=param_name,
                                                 param_range=param_range,
                                                 scoring=scoring,
                                                 fit_params=fit_params,
                                                 n_jobs=n_jobs)

    avg_train_scores = np.array([np.average(train_scores[i,:]) for i in range(train_scores.shape[0])])
    avg_test_scores = np.array([np.average(test_scores[i,:]) for i in range(test_scores.shape[0])])

    plt.rcParams['axes.labelsize'] = 15
    plt.rcParams['axes.titlesize'] = 18
    fig, ax = plt.subplots(figsize=(17,6))
    ax.plot(param_range, avg_train_scores, param_range, avg_test_scores, param_range, np.abs(avg_train_scores - avg_test_scores), 'r--')
    ax.set(title=f'Validation Metrics vs {param_name} [{estimator_name}]',
        xlabel=param_name,
        ylabel='F2 Score',
        ylim=(0,1))
    ax.legend(['Train', 'Validate', 'Deviation'])
    plt.show()

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')


def plot_label_ratios(X, y, category_name):
    target_name = y.name
    sliced_cols = [col for col in X.columns if col.startswith(category_name)]
    df = pd.concat([X[sliced_cols], y], axis=1)
    ratios_dict = {}
    for col_name in sliced_cols:
        level_name = col_name.split('_')[-1]
        ratio = df.loc[(df[col_name] == 1) & (df[target_name] == 1)].shape[0] / df.loc[(df[col_name] == 1)].shape[0]
        ratios_dict[level_name] = ratio

    levels = []
    ratios = []
    for level, ratio in sorted(ratios_dict.items(), key=lambda x: x[1]):
        levels.append(level)
        ratios.append(ratio)

    fig, ax = plt.subplots(figsize=(7,7))
    ax.bar(levels, ratios)
    ax.set(title=f"Positive Label Ratios [{category_name}]", ylim=(0,1))
    ax.yaxis.set_major_locator(MultipleLocator(base=0.1))
