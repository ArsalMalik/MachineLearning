from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


def randomSearch(classifier, parameters, cv, n_iter, train_x, train_y):
    print("***** Random Search *****")
    print("Cross-Validation:{0} and number of iterations:{1}".format(cv, n_iter))

    scores = ['accuracy']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        if (score == 'accuracy'):
            scoring_method = score
        else:
            scoring_method = score + '_micro'

        clf = RandomizedSearchCV(classifier, param_distributions=parameters, cv=cv, scoring=scoring_method, n_iter=n_iter)
        clf.fit(train_x, train_y)

        print("Best parameters and scores set found on development set:")
        # print(self.clf.best_estimator_)
        print(clf.best_params_)
        print(clf.best_score_)
        print()
        # print("Random scores on development set:")
        # means = clf.cv_results_['mean_test_score']
        # stds = clf.cv_results_['std_test_score']
        # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        #     print("%0.3f (+/-%0.03f) for %r"
        #           % (mean, std * 2, params))
        # print()

        return clf.best_params_


def gridSearch(classifier, parameters, cv, train_x, train_y):
    print("***** Grid Search *****")
    print("Cross-Validation: ",cv)

    scores = ['accuracy']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        if (score == 'accuracy'):
            scoring_method = score
        else:
            scoring_method = score + '_micro'

        clf = GridSearchCV(classifier, param_grid=parameters, cv=cv, scoring=scoring_method)
        clf.fit(train_x, train_y)

        print("Best parameters and scores set found on development set:")
        # print(self.clf.best_estimator_)
        print(clf.best_params_)
        print(clf.best_score_)
        print()
        # print("Grid scores on development set:")
        # means = clf.cv_results_['mean_test_score']
        # stds = clf.cv_results_['std_test_score']
        # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        #     print("%0.3f (+/-%0.03f) for %r"
        #           % (mean, std * 2, params))
        # print()

        return clf.best_params_

