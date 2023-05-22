import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.manifold import TSNE

plt.style.use('plotstyle.txt')
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']


def read_data(filepath):
    X = []
    y = []
    with open(filepath, 'r') as handle:
        lines = handle.readlines()
        features = [txt.strip() for txt in lines[0].strip().split(',')[4:]]
        for line in lines[1:]:
            tokens = line.strip().split(',')
            X.append([float(tok) for tok in tokens[4:]])
            y.append(int(tokens[3]))

    return X, y, features


# evaluate a model with a given number of repeats
def evaluate_model(X, y, model, test_size, repeats):
    scores = []
    for r in range(repeats):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_transformed = scaler.transform(X_train)
        clf = model.fit(X_train_transformed, y_train)
        X_test_transformed = scaler.transform(X_test)
        scores.append(clf.score(X_test_transformed, y_test))

    return scores


def find_best_model(X, y) -> GridSearchCV:
    # Define a pipeline to search for the best combination of PCA truncation
    # and classifier regularization.
    scaler = preprocessing.StandardScaler()
    # set the tolerance to a large value to make the example faster
    svc = SVC()
    pipe = Pipeline(steps=[('scaler', scaler), ('svc', svc)])

    # defining parameter range
    tuned_parameters = [{'svc__kernel': ['rbf'], 'svc__gamma': [1e-3, 1e-4], 'svc__C': [1, 10, 100, 1000]},
                        {'svc__kernel': ['linear'], 'svc__C': [1, 10, 100, 1000]},
                        {'svc__kernel': ['poly'], 'svc__degree': [2, 3]}]

    clf = GridSearchCV(estimator=pipe, param_grid=tuned_parameters, scoring='f1', n_jobs=-1, )
    clf.fit(X, y)
    return clf


def find_best_model_fixed_training_set(X, y):
    # test
    n_repeat = 1000
    scaler = preprocessing.StandardScaler()
    training_set_size = 0.7

    # result and log files
    best_param_map_count = {}
    confusion_train = np.zeros(shape=(2, 2), dtype=int)
    confusion_test = np.zeros(shape=(2, 2), dtype=int)
    scores_train = []
    scores_test = []
    for r in range(n_repeat):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=training_set_size)

        # find best model by GridSearchCV
        clf = find_best_model(X_train, y_train)
        if str(clf.best_params_) not in best_param_map_count:
            best_param_map_count[str(clf.best_params_)] = 1
        else:
            best_param_map_count[str(clf.best_params_)] += 1

        scaler.fit(X_train)
        X_train_transformed = scaler.transform(X_train)
        clf.best_estimator_.fit(X_train_transformed, y_train)
        X_test_transformed = scaler.transform(X_test)

        confusion_train = confusion_train + confusion_matrix(y_train, clf.predict(X_train_transformed))
        confusion_test = confusion_test + confusion_matrix(y_test, clf.predict(X_test_transformed))

        scores_train.append(clf.score(X_train_transformed, y_train))
        scores_test.append(clf.score(X_test_transformed, y_test))

    print('Training:Test = %d:%d split. Tested %d times. Randomly shuffled the dataset every time.' % (
        int(training_set_size * 10), int(10 - training_set_size * 10), n_repeat))
    print('[Training set] Average f1 score = %.3f, std = %.3f' % (
        float(np.mean(scores_train)), float(np.std(scores_train))))
    print('[Training set] Confusion Matrix:')
    print(confusion_train)
    print('[Test set] Average f1 score = %.3f, std = %.3f' % (float(np.mean(scores_test)), float(np.std(scores_test))))
    print('[Test set] Confusion Matrix:')
    print(confusion_test)

    list_best_param_count = list(best_param_map_count.items())
    list_best_param_count.sort(key=lambda x: x[1], reverse=True)
    print('===== Best Params ======')
    for i, pc in enumerate(list_best_param_count):
        print('#%02d: %s, %d' % (i + 1, str(pc[0]), pc[1]))

    return confusion_test


def investigate_linear_svm_coef(X, y, features):
    # test
    n_repeat = 1000
    train_size = 0.7
    scaler = preprocessing.StandardScaler()
    model = SVC(kernel='linear')

    result_file = open('results/result_csv/result_linear_coef.csv', 'w')
    result_file.write('Coef name,Coef\n')

    coefs = []
    for r in range(n_repeat):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
        scaler.fit(X_train)
        X_train_transformed = scaler.transform(X_train)
        model.fit(X_train_transformed, y_train)
        coefs.append(model.coef_)

    coefs = np.array(coefs)
    for i in range(len(features)):
        coef_rms = np.sqrt(np.mean(coefs[:, 0, i] ** 2))
        result_file.write('%s,%f\n' % (features[i], coef_rms))


def feature_selection_by_anova(X, y, features):
    n_repeats = 1000
    training_set_size = 0.7
    # #############################################################################
    # Plot the cross-validation score as a function of number of features
    ks = list(range(1, len(features) + 1))
    score_means_rbf = list()
    score_stds_rbf = list()
    score_means_lin = list()
    score_stds_lin = list()
    feature_selector = SelectKBest(f_classif)
    scaler = StandardScaler()
    svc_rbf = SVC(kernel='rbf', gamma=0.001, C=10)
    svc_lin = SVC(kernel='linear', gamma='auto')

    local_max_k = 0
    for k in ks:
        feature_selector.set_params(k=k)
        scores_rbf = []
        scores_lin = []
        for _ in range(n_repeats):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=training_set_size)
            feature_selector.fit(X_train, y_train)
            X_tn_1 = feature_selector.transform(X_train)
            X_tt_1 = feature_selector.transform(X_test)
            scaler.fit(X_tn_1)
            X_tn_2 = scaler.transform(X_tn_1)
            X_tt_2 = scaler.transform(X_tt_1)
            svc_rbf.fit(X_tn_2, y_train)
            svc_lin.fit(X_tn_2, y_train)
            scores_rbf.append(f1_score(y_test, svc_rbf.predict(X_tt_2)))
            scores_lin.append(f1_score(y_test, svc_lin.predict(X_tt_2)))

        score_means_rbf.append(np.mean(scores_rbf))
        score_stds_rbf.append(np.std(scores_rbf))
        score_means_lin.append(np.mean(scores_lin))
        score_stds_lin.append(np.std(scores_lin))

        # find the local maximum
        if k == local_max_k + 1:
            if k == 1 or (score_means_rbf[-1] > score_means_rbf[-2] and score_means_lin[-1] > score_means_lin[-2]):
                local_max_k = k

    # draw the plot
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ks, score_means_rbf, label='RBF', c=CB_color_cycle[0], )
    ax.plot(ks, score_means_lin, label='Linear', c=CB_color_cycle[4], )
    plt.vlines(12, min(score_means_rbf + score_means_lin), max(score_means_rbf + score_means_lin) + 0.01,
               colors='k', linestyles='dashed')
    #plt.vlines(local_max_k, min(score_means_rbf + score_means_lin), max(score_means_rbf + score_means_lin) + 0.01,
    #           colors='k', linestyles='dashed')
    list_xticks = [1, 5, 10, 15, 20, 25, 27]
    if local_max_k not in list_xticks:
        list_xticks.append(local_max_k)
        list_xticks.sort()
    plt.xticks(list_xticks)
    range_yticks = list(np.arange(0.8, max(score_means_rbf) + 0.02, 0.02))
    plt.yticks(range_yticks)
    plt.xlim((0, 27))
    plt.ylim((range_yticks[0], range_yticks[-1]))
    plt.xlabel('Number of features selected')
    plt.ylabel('SVM Accuracy Score')
    plt.tight_layout()
    plt.savefig('results/result_png/result_feature_selection.png')
    plt.savefig('results/result_pdf/result_feature_selection.pdf')
    plt.close()


def feature_selection_by_anova_simple(X, y, features):
    f_stats, p_values = f_classif(X, y)
    feature_table = [(name, f, p) for name, f, p in zip(features, f_stats, p_values)]
    feature_table.sort(key=lambda x: x[2])

    result_filepath = 'results/result_csv/result_feature_anova.csv'
    with open(result_filepath, 'w') as h:
        h.write("Rank,Feature name,F-statistics,P-value\n")
        for i, x in enumerate(feature_table):
            h.write("%d,%s,%f,%e\n" % (i + 1, *x))


def pca_plot(X, y, features):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    label_map = {0: 'Do not grow as expected', 1: 'Develop normally'}
    label = np.array([label_map[_y] for _y in y])

    # pca
    pca = PCA(n_components=len(features))
    X_pca = pca.fit_transform(X_scaled)

    # plot data
    pc_1 = X_pca[:, 0]
    pc_2 = X_pca[:, 1]
    df = pd.DataFrame(data={'PC1': pc_1, 'PC2': pc_2, 'label': label})
    sns.scatterplot(data=df, x="PC1", y="PC2", hue='label', palette=['#0073CF', '#BF1932'], )
    plt.savefig('results/result_png/pca_plot.png')
    plt.savefig('results/result_pdf/pca_plot.pdf')
    plt.close()

    # save pca result to csv file
    csv_filepath = 'results/result_csv/pca_coefs.csv'
    with open(csv_filepath, 'w') as h:
        h.write('Organoid_ID,Label,' + ','.join(['PC_%d' % i for i in range(1, X_pca.shape[1] + 1)]) + '\n')
        for i in range(1, X_pca.shape[0] + 1):
            h.write('%d,%s,' % (i, label[i - 1]) + ','.join([str(v) for v in X_pca[i - 1]]) + '\n')


def plot_confusion_matrix(matrix):
    confusion_matrix = np.array(matrix)
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix)
    plt.figure()
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='.03f')
    plt.xticks([0.5, 1.5], ['Do not grow as expected', 'Develop normally'])
    plt.xlabel('Prediction')
    plt.yticks([0.5, 1.5], ['Do not grow as expected', 'Develop normally'], va='center')
    plt.ylabel('Ground-truth')
    plt.tight_layout()
    plt.savefig('results/result_png/confusion_matrix.png', transparent=True)
    plt.savefig('results/result_pdf/confusion_matrix.pdf', transparent=True)
    plt.close()


def tsne_plot(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    label_map = {0: 'Do not grow as expected', 1: 'Develop normally'}
    label = np.array([label_map[_y] for _y in y])

    # t-SNE
    tsne = TSNE(n_components=2, random_state=0, perplexity=10, init='pca', method='exact', learning_rate=10)
    X_tsne = tsne.fit_transform(X_scaled)

    # plot data
    tsne_1 = X_tsne[:, 0]
    tsne_2 = X_tsne[:, 1]
    df = pd.DataFrame(data={'t-SNE Component 1': tsne_1, 't-SNE Component 2': tsne_2, 'label': label})
    sns.scatterplot(data=df, x="t-SNE Component 1", y="t-SNE Component 2", hue='label', palette=['#0073CF', '#BF1932'])
    plt.savefig('results/result_png/tSNE_plot.png')
    plt.savefig('results/result_pdf/tSNE_plot.pdf')
    plt.close()


def main():
    # read data
    X, y, features = read_data('organoid_data.csv')

    os.makedirs('results/result_csv', exist_ok=True)
    os.makedirs('results/result_png', exist_ok=True)
    os.makedirs('results/result_pdf', exist_ok=True)

    # pca plots
    pca_plot(X, y, features)

    # t-SNE plots
    tsne_plot(X, y)

    # best model
    confusion_test = find_best_model_fixed_training_set(X, y)

    # feature importance by linear svm coefficient
    investigate_linear_svm_coef(X, y, features)

    # feature selection by ANOVA analysis
    feature_selection_by_anova_simple(X, y, features)
    feature_selection_by_anova(X, y, features)

    # confusion matrix
    plot_confusion_matrix(confusion_test)


if __name__ == '__main__':
    main()
