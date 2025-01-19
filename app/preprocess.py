# Project : OC Parcours Data Scientist - P8 - Concevez un dashboard de credit scoring
# Author : Adeline Le Ray
# Date : January 2025
# Content : Load data from S3 bucket and preprocess data (cleaning, feature engineering) 

import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import streamlit as st 
import gc
from contextlib import contextmanager
import time
import boto3
from io import StringIO


class Preprocess():
    def __init__(self, debug = False):
        """!
        @brief Initialize the Preprocess class with optional debug mode.

        @param debug (bool): If True, the class will process only a limited number of rows
                             and print debugging information.
        """
        self.debug = debug
        self.AWS_REGION = "eu-west-3"
        self.BUCKET_NAME = "p8bucket"
        self.num_rows = 10000 if self.debug else None

    def _log(self, message):
        """
        @brief Log a message if debug is True.

        @param message: (str) Message to print.
        """
        if self.debug:
            try:
                st.write(message)
            except Exception as e:
                print(f"Streamlit log error: {e}")    
    

    def load_table(self, file_key):
        self._log(f"Chargement du fichier : {file_key}")
        try:
            # Initialize S3 client (uses default credential provider chain)
            s3_client = boto3.client('s3', region_name=self.AWS_REGION)

            # Fetch the object from S3
            response = s3_client.get_object(Bucket=self.BUCKET_NAME, Key=file_key)

            # Read the content of the file
            file_content = response['Body'].read().decode('utf-8')
            
            # Convert the file content to a pandas DataFrame
            data = pd.read_csv(StringIO(file_content), nrows = self.num_rows)
            return data
        
        except Exception as e:
            st.error(f"Error reading file from S3: {e}")
            return None

    def encode_categorical_columns(self, df):
        """!
        @brief Encodes categorical columns with 2 or fewer unique values using Label Encoding
        and applies One-Hot Encoding for the rest of the categorical columns.
        
        @param df (pd.DataFrame): The input DataFrame to encode.
        @returns : df (pd.DataFrame), le_col (list), encoded_col (list): The encoded DataFrame, 
                   the label encoded columns, and the final list of encoded columns.
        """
        # Create a label encoder object
        le = LabelEncoder()
        le_count = 0
        le_col = []
        one_hot_encoded_col = []

        # Iterate through the columns
        for col in df.columns:
            if df[col].dtype == 'object':
                # If 2 or fewer unique categories, apply Label Encoding
                if len(list(df[col].unique())) <= 2:
                    # Fit_transform data
                    df[col] = le.fit_transform(df[col])
                    
                    # Keep track of how many columns were label encoded and their names
                    le_count += 1
                    le_col.append(col)
                else:
                    # Track columns that will be one-hot encoded
                    one_hot_encoded_col.append(col)
        
        # Save original column names before one-hot encoding
        original_cols = df.columns.tolist()

        # One-hot encoding of remaining categorical variables
        df = pd.get_dummies(df)

        # Get new column names after one-hot encoding
        new_encoded_cols = [col for col in df.columns if col not in original_cols]

        # Combine label encoded and one-hot encoded columns
        encoded_cols = le_col + new_encoded_cols

        # Return the DataFrame and the final list of encoded columns
        return df, encoded_cols


    @contextmanager
    def timer(self, title):
        """!
        @brief A context manager to measure the execution time of a code block.

        @param title (str): Description of the code block being timed.
        """
        t0 = time.time()
        yield
        elapsed_time = time.time() - t0
        self._log(f"{title} - terminé en {elapsed_time:.0f}s")

    # Preprocess application_train.csv
    def application_train(self):
        """!
        @brief Preprocess the application_train.csv dataset. Create new features and clean specific columns.

        @return pd.DataFrame: The preprocessed application_train DataFrame.
        """
        # Load data
        df = self.load_table('application_train.csv')

        # Encoding for categorical columns
        df, _ = self.encode_categorical_columns(df)

        # NaN values for DAYS_EMPLOYED: 365.243 -> nan
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace({365243: np.nan})

        # Some simple new features (percentages)
        df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
        df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
        df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

        # # Filter on relevant features
        # df_filtered = df[['SK_ID_CURR',
        #                   'EXT_SOURCE_2', 
        #                   'EXT_SOURCE_3', 
        #                   'DAYS_EMPLOYED',
        #                   'DAYS_BIRTH', 
        #                   'DAYS_REGISTRATION',
        #                   'REGION_POPULATION_RELATIVE', 
        #                   'DAYS_ID_PUBLISH', 
        #                   'DAYS_LAST_PHONE_CHANGE',
        #                   'AMT_ANNUITY', 
        #                   'TOTALAREA_MODE',
        #                   'ANNUITY_INCOME_PERCENT',
        #                   'INCOME_PER_PERSON', 
        #                   'PAYMENT_RATE']]

        gc.collect()
        return df

    # Preprocess bureau.csv and bureau_balance.csv
    def bureau_and_balance(self):
        """!
        @brief Preprocess bureau.csv and bureau_balance.csv datasets. Perform aggregations and merge data.

        @return pd.DataFrame: The aggregated bureau and bureau_balance data.
        """
        # Load bureau and bureau_balance
        bureau = self.load_table('bureau.csv')
        bb = self.load_table('bureau_balance.csv')
        
        # Encoding for categorical columns
        bureau, bureau_cat = self.encode_categorical_columns(bureau)
        bb, bb_cat = self.encode_categorical_columns(bb)
            
        # Bureau balance: Perform aggregations and merge with bureau.csv
        bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
        for col in bb_cat:
            bb_aggregations[col] = ['mean']
        bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
        bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
        bureau = pd.merge(bureau, bb_agg, how='left', on='SK_ID_BUREAU')
        bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
        del bb, bb_agg
        import gc
        
        # Bureau and bureau_balance numeric features
        num_aggregations = {
            'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
            'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
            'DAYS_CREDIT_UPDATE': ['mean'],
            'CREDIT_DAY_OVERDUE': ['max', 'mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
            'AMT_ANNUITY': ['max', 'mean'],
            'CNT_CREDIT_PROLONG': ['sum'],
            'MONTHS_BALANCE_MIN': ['min'],
            'MONTHS_BALANCE_MAX': ['max'],
            'MONTHS_BALANCE_SIZE': ['mean', 'sum']
        }
        
        # Bureau and bureau_balance categorical features
        cat_aggregations = {cat: ['mean'] for cat in bureau_cat}
        cat_aggregations.update({cat + "_MEAN": ['mean'] for cat in bb_cat})
        bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
        bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
        
        # Bureau: Active credits - using only numerical aggregations
        active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
        active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
        active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
        bureau_agg = bureau_agg.merge(active_agg, how='left', on='SK_ID_CURR')
        del active, active_agg
        import gc

        # Bureau: Closed credits - using only numerical aggregations
        closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
        closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
        closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
        bureau_agg = bureau_agg.merge(closed_agg, how='left', on='SK_ID_CURR')
        del closed, closed_agg, bureau
        import gc

        if self.debug:
            st.dataframe(bureau_agg.head())  
        
        # # Filter on relevant features
        # bureau_agg_filtered = bureau_agg[['ACTIVE_DAYS_CREDIT_UPDATE_MEAN',
        #                                   'CLOSED_DAYS_CREDIT_MAX',
        #                                   'ACTIVE_DAYS_CREDIT_ENDDATE_MIN',
        #                                   'BURO_DAYS_CREDIT_VAR',
        #                                   'CLOSED_DAYS_CREDIT_ENDDATE_MAX',
        #                                   'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN',
        #                                   'ACTIVE_DAYS_CREDIT_MAX',
        #                                   'CLOSED_AMT_CREDIT_SUM_MEAN',
        #                                   'CLOSED_DAYS_CREDIT_UPDATE_MEAN',
        #                                   'ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN',
        #                                   'BURO_DAYS_CREDIT_MEAN',
        #                                   'CLOSED_AMT_CREDIT_SUM_SUM',
        #                                   'BURO_DAYS_CREDIT_ENDDATE_MIN']]
        return bureau_agg


    # Preprocess previous_application.csv
    def previous_applications(self):
        # Load     
        prev = self.load_table('previous_application.csv')
        
        # Encoding for categorical columns  
        prev, cat_cols = self.encode_categorical_columns(prev)

        # Days 365.243 values -> nan
        prev['DAYS_FIRST_DRAWING'] = prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan)
        prev['DAYS_FIRST_DUE'] = prev['DAYS_FIRST_DUE'].replace(365243, np.nan)
        prev['DAYS_LAST_DUE_1ST_VERSION'] = prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan)
        prev['DAYS_LAST_DUE'] = prev['DAYS_LAST_DUE'].replace(365243, np.nan)
        prev['DAYS_TERMINATION'] = prev['DAYS_TERMINATION'].replace(365243, np.nan)
        # Add feature: value ask / value received percentage
        prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
        # Previous applications numeric features
        num_aggregations = {
            'AMT_ANNUITY': ['min', 'max', 'mean'],
            'AMT_APPLICATION': ['min', 'max', 'mean'],
            'AMT_CREDIT': ['min', 'max', 'mean'],
            'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
            'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
            'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
            'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
            'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
            'DAYS_DECISION': ['min', 'max', 'mean'],
            'CNT_PAYMENT': ['mean', 'sum'],
        }
        # Previous applications categorical features
        cat_aggregations = {}
        for cat in cat_cols:
            cat_aggregations[cat] = ['mean']
        
        prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
        prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
        # Previous Applications: Approved Applications - only numerical features
        approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
        approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
        approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
        prev_agg = prev_agg.merge(approved_agg, how='left', on='SK_ID_CURR')
        # Previous Applications: Refused Applications - only numerical features
        refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
        refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
        refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
        prev_agg = prev_agg.merge(refused_agg, how='left', on='SK_ID_CURR')
        del refused, refused_agg, approved, approved_agg, prev
        gc.collect()

        if self.debug:
            st.dataframe(prev_agg.head())

        # # Filter on relevant features
        # prev_agg_filtered = prev_agg[['PREV_APP_CREDIT_PERC_VAR',
        #                               'PREV_HOUR_APPR_PROCESS_START_MEAN',
        #                               'PREV_CNT_PAYMENT_MEAN',
        #                               'PREV_APP_CREDIT_PERC_MEAN',
        #                               'PREV_AMT_ANNUITY_MIN', 
        #                               'PREV_AMT_ANNUITY_MEAN',
        #                               'PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN',
        #                               'PREV_NAME_YIELD_GROUP_middle_MEAN',
        #                               'PREV_RATE_DOWN_PAYMENT_MEAN',
        #                               'APPROVED_AMT_DOWN_PAYMENT_MAX']]
        return prev_agg

    # Preprocess POS_CASH_balance.csv
    def pos_cash(self):
        # Load 
        pos = self.load_table('POS_CASH_balance.csv')
        
        # Encoding for categorical columns
        pos, cat_cols = self.encode_categorical_columns(pos)
        
        # Features
        aggregations = {
            'MONTHS_BALANCE': ['max', 'mean', 'size'],
            'SK_DPD': ['max', 'mean'],
            'SK_DPD_DEF': ['max', 'mean']
        }
        for cat in cat_cols:
            aggregations[cat] = ['mean']
        
        pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
        pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
        # Count pos cash accounts
        pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
        del pos
        gc.collect()

        if self.debug:
            st.dataframe(pos_agg.head())

        # # Filter on relevant features
        # pos_agg_filtered = pos_agg[['POS_NAME_CONTRACT_STATUS_Active_MEAN',
        #                             'POS_MONTHS_BALANCE_SIZE']]

        return pos_agg


    # Preprocess installments_payments.csv
    def installments_payments(self):
        # Load 
        ins = self.load_table('installments_payments.csv')
        
        # Encoding for categorical columns
        ins, cat_cols = self.encode_categorical_columns(ins)
        
        # Percentage and difference paid in each installment (amount paid and installment value)
        ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
        ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
        # Days past due and days before due (no negative values)
        ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
        ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
        ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
        ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
        # Features: Perform aggregations
        aggregations = {
            'NUM_INSTALMENT_VERSION': ['nunique'],
            'DPD': ['max', 'mean', 'sum'],
            'DBD': ['max', 'mean', 'sum'],
            'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
            'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
            'AMT_INSTALMENT': ['max', 'mean', 'sum'],
            'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
            'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
        }
        for cat in cat_cols:
            aggregations[cat] = ['mean']
        ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
        ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
        # Count installments accounts
        ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
        del ins
        gc.collect()

        if self.debug:
            st.dataframe(ins_agg.head())        

        # # Filter on relevant features
        # ins_agg_filtered = ins_agg[['INSTAL_DBD_MEAN',
        #                             'INSTAL_DAYS_ENTRY_PAYMENT_MAX',
        #                             'INSTAL_AMT_PAYMENT_MIN',
        #                             'INSTAL_DBD_SUM',
        #                             'INSTAL_DBD_MAX',
        #                             'INSTAL_DAYS_ENTRY_PAYMENT_MEAN',
        #                             'INSTAL_DAYS_ENTRY_PAYMENT_SUM',
        #                             'INSTAL_AMT_PAYMENT_MEAN',
        #                             'INSTAL_DPD_MEAN',
        #                             'INSTAL_AMT_INSTALMENT_MAX',
        #                             'INSTAL_PAYMENT_PERC_VAR']]


        return ins_agg


    # Preprocess credit_card_balance.csv
    def credit_card_balance(self):
        cc = self.load_table('credit_card_balance.csv')
        
        # Encoding for categorical columns
        cc, ccb_cat = self.encode_categorical_columns(cc)
        
        # General aggregations
        cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
        cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
        cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
        # Count credit card lines
        cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
        del cc
        gc.collect()
        return cc_agg

    def aggregate_tables(self):
        
        df = self.application_train()
        with self.timer("Process bureau and bureau_balance"):
            bureau = self.bureau_and_balance()
            df = df.join(bureau, how='left', on='SK_ID_CURR')
            del bureau
            gc.collect()
        with self.timer("Process previous_applications"):
            prev = self.previous_applications()
            df = df.join(prev, how='left', on='SK_ID_CURR')
            del prev
            gc.collect()
        with self.timer("Process POS-CASH balance"):
            pos = self.pos_cash()
            df = df.join(pos, how='left', on='SK_ID_CURR')
            del pos
            gc.collect()
        with self.timer("Process installments payments"):
            ins = self.installments_payments()
            df = df.join(ins, how='left', on='SK_ID_CURR')
            del ins
            gc.collect()
        with self.timer("Process credit card balance"):
            cc = self.credit_card_balance()
            df = df.join(cc, how='left', on='SK_ID_CURR')
            del cc
            gc.collect()

        # Replace infinite by NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        return df
    