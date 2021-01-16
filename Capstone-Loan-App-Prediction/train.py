from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory


def clean_data(data):

    # Clean and fill missing values
    # AML dataset upload auto decodes (Yes, NO) and (Y, N) to True, False
    # all column dtype has to be numeric to train in sklearn
    x_df = data.to_pandas_dataframe()
    x_df.drop("Loan_ID", inplace=True, axis=1)
    x_df['Gender'].replace({'Female': 0, 'Male': 1}, inplace=True)
    x_df['Married'].replace({False: 0, True: 1}, inplace=True)
    x_df['Dependents'].replace({'0': 0, '1': 1, '2': 2, '3+': 3}, inplace=True)
    x_df['Education'].replace({'Not Graduate': 0, 'Graduate': 1}, inplace=True)
    x_df['Self_Employed'].replace({False: 0, True: 1}, inplace=True)
    x_df['Property_Area'].replace({'Rural': 0, 'Semiurban': 1, 'Urban': 2}, inplace=True)
    x_df['Loan_Status'].replace({False: 0, True: 1}, inplace=True)
    x_df.rename(columns={'Loan_Status': 'y'},inplace=True)
    #x_df.y = x_df.y.astype(int)

    x_df.Gender=np.where(x_df.Gender.isna(),1,x_df.Gender)
    x_df.Married=np.where(x_df.Married.isna(),False,x_df.Married)
    x_df.Dependents=np.where(x_df.Dependents.isna(),0,x_df.Dependents)
    x_df.Self_Employed=np.where(x_df.Self_Employed.isna(),False,x_df.Self_Employed)
    topfrq = x_df.CoapplicantIncome.value_counts().index[0]
    x_df.CoapplicantIncome=np.where(x_df.CoapplicantIncome.isna(),topfrq,x_df.CoapplicantIncome)
    topamt = x_df.LoanAmount.value_counts().index[0]
    x_df.LoanAmount=np.where(x_df.LoanAmount.isna(),topamt,x_df.LoanAmount)
    minterm = x_df.Loan_Amount_Term.value_counts().index[-1]
    x_df.Loan_Amount_Term=np.where(x_df.Loan_Amount_Term.isna(),minterm,x_df.Loan_Amount_Term)
    x_df.Credit_History=np.where(x_df.Credit_History.isna(),0,x_df.Credit_History)
    for i in x_df.columns:
        if x_df[i].dtype in ('object','bool'):
            print(f'col: {i}')
            x_df[i] = x_df[i].astype(int)

    y_df=x_df.pop('y')

    return x_df, y_df


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run = Run.get_context()
    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    # Create TabularDataset using TabularDatasetFactory
    # from web url:
    # "https://raw.githubusercontent.com/atan4583/datasets/master/train.csv"

    wurl='https://raw.githubusercontent.com/atan4583/datasets/master/train.csv'
    ds = TabularDatasetFactory.from_delimited_files(wurl)

    x, y = clean_data(ds)
    print(f'x null chk: \n{x.isnull().sum()}\n \ny null chk: \n{y.isnull().sum()}\n')
    # Split data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=42)
    print(f'x_train null chk: \n{x_train.isnull().sum()}\n \ny_train null chk: \n{y_train.isnull().sum()}\n')
    print(f'x_test null chk: \n{x_test.isnull().sum()}\n \ny_test null chk: \n{y_test.isnull().sum()}\n')

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('outputs',exist_ok=True)
    joblib.dump(model,'outputs/model.pkl')


if __name__ == '__main__':
    main()
