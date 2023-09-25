import pandas as pd

s = pd.read_csv("./data/d-20230922.csv")

s.drop(columns=["Age", "IGRAs-CZ", "Sex"], inplace=True)

K = s.drop_duplicates(subset=["Cough", "Fever", "Hemoptysis", "CT", "IGRAs", "TBC", "TB-DNA"])

K.drop(columns=["label"], inplace=True)

counts = s.groupby(["Cough", "Fever", "Hemoptysis", "CT", "IGRAs", "TBC", "TB-DNA"])['label'].value_counts().unstack(fill_value=0)

counts.columns = ['ntb', 'ntm', 'tb']

K = pd.merge(K, counts, on=["Cough", "Fever", "Hemoptysis", "CT", "IGRAs", "TBC", "TB-DNA"], how="left")

print(K)

K.to_csv('./data/d-20230922-2.csv', index=False)