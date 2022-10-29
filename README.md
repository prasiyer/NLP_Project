## Installation ##
In order to run the notebooks in this repository, the following libraries have to be installed:
1) Pandas 0.24.2
2) Numpy 1.17.4
3) pickle
4) sqlalchemy
5) seaborn 0.9.0
6) scikit-learn 0.21.2
7) nltk

## Project Motivation ##
The purpose of this project is to create a classifier for classifying input messages. The input contains messages received in disaster zones. The classifier is used to classify these input messages into categories. In turn, the predicted category can be used to route the message to the appropriate agency. The intended benefit is prompt response to the incoming messages

## Technical Details ##
This project demonstrates:
1) Use of Pipeline to execute ML workflow
2) The workflow steps consist of 
    a) reading data and cleaning data
    b) training a classifier
    c) evaluating the trained classifier
    
## File Descriptions ##
The repository consists of 2 main folders -- Data & Code
The Data folder has:
1) 2 CSV files: disaster_messages.csv and disaster_categories.csv are the input data files. These files contain the messages received from disaster regions and the corresponding categories of the messages, respectively
2) Database file: DisasterResponse.db is a SQLite database. This database has a main table (Message_Category). This table contains the clean data [X: Tokenized message, y: Categories] used to train and evaluate the classifier

The Code folder has:
1) Data_ETL.ipynb & process_data.py: These are the Jupyter notebook and the corresponding python script for reading, cleaning and loading of the input data into a database
2) ML_NLP_Workflow.ipynb and train_classifier.py: These are the Jupyter notebook and corresponding python script for training the classifier. This script utilizes GridSearch among RandomForest and KNeighors classifiers. 
3) Model_Evaluation: This notebook analyzes the performance of the classifier. The output categories are separated into 2 sets [prominent and other] based on the frequency of their occurence in the dataset


## Acknowledgements ##
Thanks to Python open source community for creating valuable libraries used in this project. <br>
This project uses normalized dataset of truckload shipments

## License ##
Apache license
