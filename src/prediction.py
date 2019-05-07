from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


def prediction(df):
    train, test = seperate_train_and_test_data(df)
    x_train, y_train, x_test, y_test = seperate_features_and_label(train, test)
    # knn(x_train, y_train, x_test, y_test)
    # d_tree(x_train, y_train, x_test, y_test)
    svm(x_train, y_train, x_test, y_test)


def get_optimal_hyper_parameters_for_svm(x_train, y_train):
    grid_param = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10],
        'gamma': ['auto', 0.1, 1, 10]
    }
    clf = SVC()
    # search best parameters with cross validation using grid search
    gr = GridSearchCV(estimator=clf, param_grid=grid_param, scoring='accuracy', cv=5)
    gr.fit(x_train, y_train)

    print("\n\n<<Kernel Used: {} >>".format(gr.best_params_['kernel']))
    print("<<C Used: {} >>".format(gr.best_params_['C']))
    print("<<Gamma Used: {} >>".format(gr.best_params_['gamma']))
    print("<<Best Accuracy for training: {0:.2f}% >>".format(gr.best_score_ * 100))

    return gr.best_params_


def svm(x_train, y_train, x_test, y_test):
    # best_params = get_optimal_hyper_parameters_for_svm(x_train, y_train)
    best_params = {
        'kernel': 'linear',
        'C': 1,
        'gamma': 'auto'
    }

    clf = SVC(kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma'])
    clf.fit(x_train, y_train)

    pred = clf.score(x_train, y_train)
    print("<<Train Accuracy Rate: {0:.2f}%>>".format(pred * 100))

    pred = clf.score(x_test, y_test)
    print("<<Test Accuracy Rate: {0:.2f}%>>".format(pred * 100))


def d_tree(x_train, y_train, x_test, y_test):
    tree = DecisionTreeClassifier(random_state=1)
    tree.fit(x_train, y_train)

    pred = tree.score(x_train, y_train)
    print("<<Train Accuracy Rate: {0:.2f}%>>".format(pred * 100))

    pred = tree.score(x_test, y_test)
    print("<<Test Accuracy Rate: {0:.2f}%>>".format(pred * 100))


def knn(x_train, y_train, x_test, y_test):
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(x_train, y_train)

    pred = neigh.score(x_train, y_train)
    print("<<Train Accuracy Rate: {0:.2f}%>>".format(pred * 100))

    pred = neigh.score(x_test, y_test)
    print("<<Test Accuracy Rate: {0:.2f}%>>"
          .format(pred * 100))


def seperate_features_and_label(train, test):
    x_train = train.drop('quality', axis=1).values
    y_train = train['quality']
    x_test = test.drop('quality', axis=1).values
    y_test = test['quality']

    return x_train, y_train, x_test, y_test


def seperate_train_and_test_data(df):
    train = df[:4000]
    test = df[4000:]

    return train, test
