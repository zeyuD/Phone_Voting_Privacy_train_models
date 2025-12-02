import pandas as pd
data_names = ["ArmHand"]

data_name = data_names[0]
# Remove NaN values and move next row up
df = pd.read_csv("F1_avg.csv")
df1 = df.apply(lambda x: pd.Series(x.dropna().values))
df1.to_csv("F1_avg_"+data_name+".csv", index=False)
df_ = pd.read_csv("FNR_avg.csv")
df2 = df_.apply(lambda x: pd.Series(x.dropna().values))
df2.to_csv("FNR_avg_"+data_name+".csv", index=False)
