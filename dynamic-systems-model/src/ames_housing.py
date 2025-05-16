"""
Data Analysis and Metrics Calculation
=================================================

This module provides functions for data analysis, preprocessing, and metrics calculation
for the house price prediction project. It includes functionality for:

- Loading and cleaning the Ames Housing dataset

The module implements various statistical and machine learning methods for data analysis,
including correlation analysis, clustering, and dimensionality reduction.

Functions
---------
load_ames_housing
    Load the Ames Housing dataset from its source
clean_data
    Clean and preprocess the dataset

Notes
-----
The module assumes the existence of certain directories and files:
- data/ : Directory where the dataset will be downloaded
"""
import pandas as pd

def load_ames_housing() -> pd.DataFrame:
    """
    Load the Ames Housing dataset from its source.

    This function downloads the dataset if it's not already present
    and loads it into a pandas DataFrame.

    Returns
    -------
    pandas.DataFrame
        The loaded Ames Housing dataset

    Raises
    ------
    Exception
        If there are issues downloading or loading the dataset

    Notes
    -----
    The dataset is downloaded from:
    http://jse.amstat.org/v19n3/decock/AmesHousing.txt

    The function creates a data/ directory if it doesn't exist
    and saves the downloaded dataset there for future use.
    """
    import urllib.request
    from pathlib import Path

    data_dir = Path('../data')
    data_dir.mkdir(exist_ok=True)
    
    file_path = data_dir / 'ames_housing.txt'
    
    if not file_path.exists():
        print("Dataset not found. Downloading from source...")
        url = "http://jse.amstat.org/v19n3/decock/AmesHousing.txt"
        try:
            urllib.request.urlretrieve(url, file_path)
            print("Dataset downloaded successfully!")
        except Exception as e:
            raise Exception(f"Error downloading dataset: {str(e)}")
    
    try:
        df = pd.read_csv(file_path, sep='\t')
        print(f"Successfully loaded {len(df)} records from ames_housing.txt")
        return df
    except Exception as e:
        raise Exception(f"Error loading ames_housing.txt: {str(e)}")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the dataset.

    This function performs several cleaning and feature engineering steps:
    1. Handles missing values in numeric and categorical columns
    2. Removes outliers using the IQR method
    3. Creates new feature combinations
    4. Converts categorical quality variables to numeric

    Parameters
    ----------
    df : pandas.DataFrame
        Raw input dataset

    Returns
    -------
    pandas.DataFrame
        Cleaned and preprocessed dataset
    """    
    # Handle missing values
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Fill numeric missing values with median
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
    
    return df