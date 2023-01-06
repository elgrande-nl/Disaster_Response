import logging
import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO)

# ETL Pipeline Preparation

# In a Python script, process_data.py, write a data cleaning pipeline that:
# Loads the messages and categories datasets
# Merges the two datasets
# Cleans the data
# Stores it in a SQLite database


def load_data(messages_filepath, categories_filepath):
    """
    Function:
    Load data from the 2 files and merge them to one Pandas DataFrame.

    Args:
        messages_filepath (str): filepath of the messages file.
        categories_filepath (str): filepath of the categories file.

    Returns:
        df (Pandas DataFrame): A dataframe with the messages and there categories.
    """

    messages = pd.read_csv(messages_filepath)
    logging.info(
        "messages has {} rows and {} columns.".format(
            len(messages), len(messages.columns)
        )
    )
    categories = pd.read_csv(categories_filepath)
    logging.info(
        "categories has {} rows and {} columns.".format(
            len(categories), len(categories.columns)
        )
    )
    df = pd.merge(left=messages, right=categories, on="id")

    return df


def clean_data(df):
    """
    Function:
    Clean the Dataframe df and prepare for ML flow

    Args:
        df (DataFrame): A dataframe that needs to be cleanded

    Returns:
        df (DataFrame): The cleanded dataframe
    """

    # Split categories into separate category columns.
    categories = df["categories"].str.split(";", expand=True)

    # Rename the columns of `categories`
    row = categories.head(1)
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0, :]
    category_colnames = category_colnames.tolist()
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], errors="coerce")

    # Replace categories column in df with new category columns.
    df = pd.concat([df, categories], axis=1, join="inner")

    # Remove duplicates.
    logging.info(
        "There where {} duplicated rows in the DataFrame".format(df.duplicated().sum())
    )
    df.drop_duplicates(inplace=True)

    return df


def save_results_to_database(df, database_filepath):
    """
    Function:
    Save the DataFrame df in to the given database

    Args:
        df (DataFrame): The datagrame with cleaned messages and catagories.
        database_file (str): The database filename.
    """
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df.to_sql("tbl_MessagesCategories", engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        logging.info(
            "Loading data and merging files...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )

        df = load_data(messages_filepath, categories_filepath)

        logging.info("Cleaning data and prepare for ML flow...")
        df = clean_data(df)

        logging.info("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_results_to_database(df, database_filepath)

        logging.info("done.")

    else:
        logging.info(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
            "example: `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`"
        )


if __name__ == "__main__":
    main()
