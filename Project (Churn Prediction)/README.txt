=================================================

			README FILE

=================================================


Language Used:
-------------
- Python


===================================================

				How To Run

===================================================

- Unzip the folder and save it in a local directory
- Create 2 folders named "data" and "processed_data" in the directory
- Put the data files in the "data" folder
- Pre-process the members.csv dataset by running the following command,
    python Members.py
- Pre-process the transactions.csv dataset by running the following command,
    python Transactions.py
- Pre-process the user_log.csv dataset by running the following command,
    python Preprocessing_user_logs.py
- Merge all the dataset and make the training dataset by running the following command,
    python Train.py
- Merge all the dataset and make the test dataset by running the following command,
    python Test.py
- To find the best parameters run the following command,
    python SearchParameters.py
- To run the classification run the following command,
    python Model.py
