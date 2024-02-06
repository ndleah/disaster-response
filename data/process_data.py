import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets.

    Parameters:
    - messages_filepath (str): Filepath for the messages dataset.
    - categories_filepath (str): Filepath for the categories dataset.

    Returns:
    - df (DataFrame): Merged DataFrame containing messages and categories.
    """
    # Load messages dataset
    messages = pd.read_csv(messages_filepath, index_col=False)
    
    # Load categories dataset
    categories = pd.read_csv(categories_filepath, index_col=False)
    
    # Merge datasets
    df = pd.merge(messages, categories, on='id')
    
    return df


def clean_data(df):
    """
    Clean and preprocess the merged DataFrame.

    Parameters:
    - df (DataFrame): Merged DataFrame containing messages and categories.

    Returns:
    - df (DataFrame): Cleaned and preprocessed DataFrame.
    """
    # Split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Extract column names from the first row
    category_colnames = categories.iloc[0].apply(lambda x: x.split('-')[0])
    
    # Rename the columns
    categories.columns = category_colnames
    
    # Convert category values to 0 or 1
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1].astype(int)
    
    # Drop the original categories column and concatenate the new one
    df = pd.concat([df.drop('categories', axis=1), categories], axis=1)
    
    # Drop duplicates
    df = df.drop_duplicates()

    # Remove rows containing '2' in any column
    df = df[~(df == 2).any(axis=1)]
    
    return df


def save_data(df, database_filename):
    """
    Save the cleaned DataFrame to an SQLite database.

    Parameters:
    - df (DataFrame): Cleaned and preprocessed DataFrame.
    - database_filename (str): Filepath for the SQLite database.

    Returns:
    - None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_response', engine, index=False, if_exists='replace')


def main():
    """
    Main ETL (Extract, Transform, Load) process to execute data cleaning and saving.

    Parameters:
    - None

    Returns:
    - None
    """
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