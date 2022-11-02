import sys
import nltk
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


def load_data(database_filepath):
    '''
    Load data into a pandas dataframe from a table located in the input database
    Parameters:
        database_filepath (str): Path to the database file
    '''
    db_filename = 'sqlite:///' + database_filepath
    db_engine = create_engine(db_filename)
    msg_df = pd.read_sql_table('Message_Category', db_engine)
    X = msg_df['message'].values
    for column_name in msg_df.columns[~msg_df.columns.isin(['id', 'genre', 'message', 'original'])]:
        msg_df[column_name] = msg_df[column_name].astype('int')
    y = msg_df.loc[:, ~msg_df.columns.isin(['id', 'genre', 'message', 'original'])].values
    columnn_labels = msg_df.columns[~msg_df.columns.isin(['id', 'genre', 'message', 'original'])]
    return X, y, columnn_labels


def tokenize(text):
    '''
    Create tokens and then lemmatize the tokens. Useful for including in CountVectorizer
    Parameters:
        text (str): Input text to be tokenized
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for tok in tokens:
        clean_token = lemmatizer.lemmatize(tok).lower().strip()
        lemmatized_tokens.append(clean_token)
    return lemmatized_tokens


def build_model():
    '''
    Create and return a GridSearchCV instance. The input to the GridSearch consists of a Pipeline instand and with 2 parameters corresponding to the estimators
    Returns: GridSearchCV instance
    '''
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('MOClf', MultiOutputClassifier(estimator = KNeighborsClassifier()))])
    parameters = {'MOClf__estimator': [RandomForestClassifier(), KNeighborsClassifier() ]}
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Calculate accuracy, precision, recall and f1 score for the input model
    Parameters:
        trained_model: Model to be used for prediction on the test dataset
        X_test: Input for the prediction
        y_test: Ground truth or the correct prediction
        y_pred: Predicted results
        column_labels: Labels for the predicted variables
    Returns:
        trained_model_accuracy: Overall accuracy of the trained model
        accuracy_df: Dataframe containing the performance metrics on the test set
    '''
    Y_pred = model.predict(X_test)
    test_result_df = pd.DataFrame(Y_test, columns = category_names)
    pred_result_df = pd.DataFrame(Y_pred, columns = category_names)
    #column = 'related'
    #accuracy = accuracy_score(test_result_df['related'], pred_result_df['related'])
    trained_model_accuracy = model.score(X_test, Y_test)
    accuracy_df = pd.DataFrame(data = [], columns = ['Category', 'Accuracy','Precision', 'Recall', 'F1'])
    i = 0
    for column_name in category_names:
        accuracy_score_column = round(accuracy_score(test_result_df[column_name], pred_result_df[column_name]), 2)
        precision_score_column = round(precision_score(test_result_df[column_name], pred_result_df[column_name]), 2)
        recall_score_column = round(recall_score(test_result_df[column_name], pred_result_df[column_name]), 2)
        f1_score_column = round(f1_score(test_result_df[column_name], pred_result_df[column_name]), 2)
        #print(column_name, '----', accuracy_score_column)
        accuracy_df.loc[i] = [column_name, accuracy_score_column, precision_score_column, recall_score_column, f1_score_column]
        i += 1
    print('     Evaluation results by category:')
    print(accuracy_df)
    return accuracy_df, trained_model_accuracy


def save_model(model, model_filepath):
    '''
    Save the input classifier model as a pickle file
    Parameters:
        model: Trained classifier model
        model_filepath (str): Path where the model file should be saved
    '''
    #filename = 'cv_model.sav'
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()