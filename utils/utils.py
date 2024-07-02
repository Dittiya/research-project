import torch
import pandas as pd

def transform_ghg_dataset(dataset):
    """
    Transform New Zealand Greenhouse Gas Emissions 2050 Projection dataset
    """
    dataframe = pd.read_excel(dataset, sheet_name=0, header=10)
    dataframe = dataframe.T
    dataframe.columns = dataframe.iloc[0]
    dataframe = dataframe[1:]
    dataframe = dataframe.reset_index().rename(columns={"index":"Year"})
    dataframe = dataframe.rename_axis(None, axis=1)
    dataframe = dataframe.astype("float32")

    return dataframe

def create_dataset(dataset, lookback):
    """
    Create LSTM dataset

    returns tensor(batch, lookback, features), tensor(target)
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)

    return torch.tensor(X), torch.tensor(y)