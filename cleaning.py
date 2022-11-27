import pandas as pd
from datetime import datetime
import glob
import os
import warnings
import json
import re
warnings.filterwarnings("ignore")

def read_data(path):
    '''This function reads in data from multiple .csv files
    within a folder given by the path variable. The function
    returns the data in a pandas dataframe.
    Note: all .csv files need to have the same columns and
    identical column names.
    '''
    #collect all files within the folder given by path
    all_files = glob.glob(os.path.join(path , "*.csv"))

    #read in all files
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    #put all files into final dataframe
    df_final = pd.concat(li, axis=0, ignore_index=True)

    return df_final


def drop_duplicate(df, feature="id"):
    '''This function deletes duplicates from a given dataframe.
    The column by which the duplicates are identified can be passed
    to the function as a second argument. The cleaned dataframe is returned.
    '''
    #first, we check for duplicates
    entries = df[feature].count()
    unique_entries = df[feature].nunique()
    print("Total number of entries:", entries)
    print("Number of unique entries:", unique_entries)
    print("Number of rows that will be deleted:", entries - unique_entries)
    #remove duplicate entries
    df.drop_duplicates(subset=[feature],inplace=True)

    return df

def clean_data(df_kick):
    '''This is a function written only for the kickstarter
    machine learning project. All column names are hard-coded.
    The function removes features from the dataframe that are not
    needed for the project. It also transforms all timestamp columns
    to datatime, removes some rows with missing data, creates a goal_usd
    column and transforms the state column to values of 0 and 1.
    '''
    #first drop redundant, unnecessary or almost empty collumns
    columns_drop = ['state_changed_at','launched_at','deadline',"location","name","pledged","profile",'is_starrable', 'id', 'photo', 'blurb', 'currency_symbol', 'fx_rate', 'currency_trailing_code', 'usd_type', 'urls', 'source_url', 'converted_pledged_amount','friends','is_starred','is_backing','permissions','current_currency']

    df_kick = df_kick.drop(columns_drop, axis=1)

    # Changing format of the timestamp columns
    timestamp_columns = ['created_at']

    for col in timestamp_columns:
        df_kick[col] = df_kick[col].apply(datetime.fromtimestamp)

    #create goal_usd...
    df_kick["goal_usd"] = df_kick["goal"] * df_kick["static_usd_rate"]
    df_kick["category_clean"] = [re.sub("/.*","", json.loads(entry)["slug"]) for entry in df_kick.category ]
    df_kick["creator_clean"] = [json.loads(re.sub(",.*" , "}", entry))["id"] for entry in df_kick.creator ]

    #...and the redundant information on the foreign currency
    del_columns = ["goal", "currency", "static_usd_rate"]
    df_kick = df_kick.drop(del_columns, axis=1) 

    #drop rows where state is suspended or live
    df_kick = df_kick.drop(df_kick[df_kick["state"]=="suspended"].index)
    df_kick = df_kick.drop(df_kick[df_kick["state"]=="live"].index)
    df_kick.reset_index(drop=True, inplace=True)

    #put "canceled" into the "failed" category
    df_kick.state = df_kick.state.replace("canceled","failed")

    #transform state to values of 0 and 1
    df_kick.state = df_kick.state.replace("successful", 1)
    df_kick.state = df_kick.state.replace("failed", 0)
    df_kick.state.unique()

    return df_kick