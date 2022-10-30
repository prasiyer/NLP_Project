import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Load data from two csv files into pandas dataframes. Merge the two individual dataframes and return the merged dataframe
    Parameters:
        messages_filepath (str): Path to the messages file
        categories_filepath (str): Path to the categories file
    Returns: Dataframe 
    '''
    msg_df = pd.read_csv(messages_filepath)
    category_df = pd.read_csv(categories_filepath)
    consol_df = msg_df.merge(category_df, left_on = 'id', right_on = 'id')
    return consol_df
    

def clean_data(consol_df):
    '''
    Tranform the input dataframe:
        1) Split the category values separated by ";"  and expand the values into individual columns
        2) Convert the category column type to Integer
        3) Concatenate the dataframe with the individual categories to the original messages dataframe
    Parameters:
        consol_df (Dataframe): Input data to be transformed
    Returns: Dataframe 
    '''
    categories = consol_df['categories'].str.split(pat = ';', expand = True)
    cat_names = categories[0:1].apply(lambda x: x.str[:-2]).astype(str)
    for column in categories.columns:
        categories[column] = categories[column].str[-1].astype(int)
        categories[column] = categories[column].apply(lambda x: 1 if x > 1 else x)
    msg_cat_df = pd.concat([consol_df, categories], axis = 1)
    msg_cat_df.drop(columns = ['categories'], axis = 1, inplace = True)
    msg_cat_clean_df = msg_cat_df.drop_duplicates()
    return msg_cat_clean_df


def save_data(df, database_filename):
    '''
    Save the input dataframe as table in the specified database
    Parameters:
        df (Dataframe): Input data to be saved
        database_filename (str): Path to the categories file
    Returns: Dataframe 
    '''
    #db_filename = 'sqlite:////data2/home/prasannaiyer/Projects/NLP_Project/Data/DisasterResponse1.db'
    db_filename = 'sqlite:///' + database_filename
    engine = create_engine(db_filename)
    df.to_sql('Message_Category', engine, index=False, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()