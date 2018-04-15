=================================================

			README FILE

=================================================

Dataset Used:
------------

- Car Dataset
  https://archive.ics.uci.edu/ml/datasets/Car+Evaluation

Language Used:
-------------

- Python 

Packages Used:
-------------

=> Scikit-Learn
	- sklearn.model_Selection.cross_val_score

* for Decision tree,
	- sklearn.tree
	
* for Deep Learning,
	- sklearn.neural_network.MLPClassifier

* for Gradient Boosting,
	- sklearn.ensemble.GradientBoostingClassifier

* for KNN,
	- sklearn.neighbors.KNeighborsClassifier

* for Naive Bayes,
	- sklearn.naive_bayes.BernoulliNB

* for Neural Net,
	- sklearn.neural_network.MLPClassifier

* for Perceptron,
	- sklearn.linear_model.Perceptron

* for Random Forest,
	- sklearn.ensemble.RandomForestClassifier

* for SVM,
	- sklearn.svm.SVC

* for Logistic Regression,
	- sklearn.linear_model.LogisticRegression

* for Bagging,
	- sklearn.ensemble.BaggingClassifier

* for Adaboost,
	- sklearn.ensemble.AdaBoostClassifier


===================================================

				How To Run

===================================================

- Unzip the folder and save it in a local directory
- Open the command prompt from inside the source code folder
- Invoke the program by running the following command,
	- Python.exe Main.py "complete path of the Car dataset"

- The Main.py will look for best parameters for every classifier 
  using randomSearch() and will output the accuracy and recall for 
  each of them by running them with the best parameters found.