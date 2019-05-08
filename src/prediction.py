from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

knn_params = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

decision_tree_params = {
    'min_samples_split': [8, 12],
    'max_depth': [5, 10, 80]
}
svm_params = {
    'kernel': ['rbf', 'linear'],
    'C': [0.1, 1, 10],
    'gamma': ['auto', 0.01, 0.1, 1, 10]
}

random_forest_params = {
    'max_depth': [80, 100],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 5],
    'min_samples_split': [8, 12],
    'n_estimators': [100, 200],
    'random_state': [None, 1, 10, 100]
}

logistic_regression_params = {
    'C': np.logspace(0, 4, 10),
    'penalty': ['l1', 'l2']  # l1 lasso l2 ridge
}


def prediction(df):
    train, test = seperate_train_and_test_data(df)
    x_train, y_train, x_test, y_test = seperate_features_and_label(train, test)
    knn(x_train, y_train, x_test, y_test)
    decision_tree(x_train, y_train, x_test, y_test)
    svm(x_train, y_train, x_test, y_test)
    random_forest(x_train, y_train, x_test, y_test)
    logistic_regression(x_train, y_train, x_test, y_test)


def logistic_regression(x_train, y_train, x_test, y_test):
    print('\n\nlogistic regression')
    # params = get_optimal_hyper_parameters(x_train, y_train, 'lr')
    params = {
        'C': 464.15888336127773,
        'penalty': 'l2'
    }
    log_clf = LogisticRegression(**params)
    training_and_test(log_clf, x_train, y_train, x_test, y_test)


def random_forest(x_train, y_train, x_test, y_test):
    print('\n\nrandom forest')
    # params = get_optimal_hyper_parameters(x_train, y_train, 'rf')
    params = {
        'max_depth': 80,
        'max_features': 3,
        'min_samples_leaf': 3,
        'min_samples_split': 8,
        'n_estimators': 200,
        'random_state': 100
    }
    rf = RandomForestClassifier(**params)
    training_and_test(rf, x_train, y_train, x_test, y_test)


def svm(x_train, y_train, x_test, y_test):
    print('\n\nsvm')
    # params = get_optimal_hyper_parameters(x_train, y_train, svm)
    params = {
        'kernel': 'linear',
        'C': 10,
        'gamma': 'auto'
    }
    clf = SVC(**params)
    training_and_test(clf, x_train, y_train, x_test, y_test)


def decision_tree(x_train, y_train, x_test, y_test):
    print('\n\ndecision_tree')
    # params = get_optimal_hyper_parameters(x_train, y_train, 'dt')
    params = {
        'max_depth': 5,
        'min_samples_split': 8
    }
    tree = DecisionTreeClassifier(**params)
    training_and_test(tree, x_train, y_train, x_test, y_test)


def knn(x_train, y_train, x_test, y_test):
    print('\n\nknn')
    params = get_optimal_hyper_parameters(x_train, y_train, 'knn')
    params = {
        'metric': 'manhattan',
        'n_neighbors': 5,
        'weights': 'distance'
    }
    neigh = KNeighborsClassifier(**params)
    training_and_test(neigh, x_train, y_train, x_test, y_test)


def seperate_features_and_label(train, test):
    x_train = train.drop('quality', axis=1).values
    y_train = train['quality'].values
    x_test = test.drop('quality', axis=1).values
    y_test = test['quality'].values

    return x_train, y_train, x_test, y_test


def seperate_train_and_test_data(df):
    eighty_percent = int(len(df)*80/100)
    train = df[:eighty_percent]
    test = df[eighty_percent:]

    print(len(test[test['quality'] == 'high']))
    print(len(test[test['quality'] == 'low']))

    return train, test


def training_and_test(clf, x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)

    pred = clf.score(x_train, y_train)
    print("Train Accuracy Rate: {0:.2f}%".format(pred * 100))

    pred = clf.score(x_test, y_test)
    print("Test Accuracy Rate: {0:.2f}%".format(pred * 100))

    y_pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)


def get_optimal_hyper_parameters(x_train, y_train, classification):
    if classification == 'knn':
        grid_param = knn_params
        clf = KNeighborsClassifier()
    elif classification == 'dt':
        grid_param = decision_tree_params
        clf = DecisionTreeClassifier()
    elif classification == 'svm':
        grid_param = svm_params
        clf = SVC()
    elif classification == 'rf':
        grid_param = random_forest_params
        clf = RandomForestClassifier()
    elif classification == 'lr':
        grid_param = logistic_regression_params
        clf = LogisticRegression()
    else:
        print('Wrong Classification')
        exit()

    gr = GridSearchCV(estimator=clf, param_grid=grid_param, scoring='accuracy', cv=5, n_jobs=-1)
    gr.fit(x_train, y_train)

    print(gr.best_params_)
    print("Best Accuracy for training: {0:.2f}%\n".format(gr.best_score_ * 100))

    return gr.best_params_
