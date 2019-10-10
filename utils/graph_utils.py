import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

HISTORY_PATH="history/history.csv"

if __name__ == "__main__":
    dataframe = pd.read_csv(HISTORY_PATH, delimiter=',')
    models =dataframe['model']
    loss = dataframe ['loss']
    acc = dataframe ['acc']
    val_loss = dataframe['val_loss']
    val_acc = dataframe['val_acc']
    epoch = dataframe.iloc[:,5]
    data = dataframe.iloc[:, 1:5]
    import pdb; pdb.set_trace()
    # ax = sns.lineplot(x=epoch, y=loss, hue=models, data=dataframe)
    # plt.figure()
    # ax = sns.lineplot(x=epoch, y=acc, hue=models, data=dataframe)
    # plt.figure()
    # ax = sns.lineplot(x=epoch, y=val_loss, hue=models, data=dataframe)
    # plt.figure()
    # ax = sns.lineplot(x=epoch, y=val_acc, hue=models, data=dataframe)
    # plt.show()