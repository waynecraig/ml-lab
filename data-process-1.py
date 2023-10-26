import pandas as pd

s_ntb = pd.read_csv('./data/结核数据/非-TB-Table 1.csv')
s_ntm = pd.read_csv('./data/结核数据/NTM-Table 1.csv')
s_tb = pd.read_csv('./data/结核数据/TB-Table 1.csv')

s_ntb['label'] = 'ntb'
s_ntm['label'] = 'ntm'
s_tb['label'] = 'tb'

s = pd.concat([s_ntb, s_ntm, s_tb], axis=0)

s.dropna()

s.reset_index(drop=True, inplace=True)

s.drop(columns=['住院号'], inplace=True)

s.rename(
  columns={
    "年龄": "Age",
    "性别": "Sex",
    "咳嗽": "Cough",
    "发热": "Fever",
    "咯血": "Hemoptysis",
    "肺部CT": "CT",
    "TBC涂片": "TBC",
  },
  inplace=True,
)

s["Sex"] = s.apply(lambda row: 1 if row["Sex"] == "女" else 0, axis=1)
s["Cough"] = s.apply(lambda row: 1 if row["Cough"] == "有" else 0, axis=1)
s["Fever"] = s.apply(lambda row: 1 if row["Fever"] == "有" else 0, axis=1)
s["Hemoptysis"] = s.apply(lambda row: 1 if row["Hemoptysis"] == "有" else 0, axis=1)
s["CT"] = s.apply(lambda row: 1 if row["CT"] == "CT-阳性" else 0, axis=1)
s["IGRAs"] = s.apply(lambda row: 1 if row["IGRAs"] == "阳性" else 0, axis=1)
s["TBC"] = s.apply(lambda row: 1 if row["TBC"] == "阳性" else 0, axis=1)
s["TB-DNA"] = s.apply(lambda row: 1 if row["TB-DNA"] == "阳性" else 0, axis=1)

s.to_csv('./data/d-20230922-1.csv', index_label="id")

s['label'] = s.apply(lambda row: 1 if row['label'] == 'tb' else 0, axis=1)

s.to_csv('./data/d-20230922-2.csv', index_label="id")