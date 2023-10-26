import pandas as pd

cols = [
  "住院号",
  "性别",
  "年龄",
  "咳嗽",
  "发热",
  "咯血",
  "肺部CT",
  "IGRAs-CZ",
  "IGRAs",
  "TBC涂片",
  "TB-DNA",
]


s_aptb = pd.read_csv('./data/结核数据20231016/APTB-Table 1.csv', usecols=cols)
s_cdptb = pd.read_csv('./data/结核数据20231016/CDPTB-Table 1.csv', usecols=cols)
s_itb = pd.read_csv('./data/结核数据20231016/ITB-Table 1.csv', usecols=cols)
s_n_tpd = pd.read_csv('./data/结核数据20231016/N-TPD-Table 1.csv', usecols=cols)

s_aptb['label'] = 'APTB'
s_cdptb['label'] = 'CDPTB'
s_itb['label'] = 'ITB'
s_n_tpd['label'] = 'N-TPD'

s = pd.concat([s_aptb, s_cdptb, s_itb, s_n_tpd], axis=0)

s.dropna()

s.reset_index(drop=True, inplace=True)

s.drop(columns=['住院号'], inplace=True)

s.rename(
  columns={
    "年龄": "Age",
    "性别": "Gender",
    "咳嗽": "Cough",
    "发热": "Fever",
    "咯血": "Hemoptysis",
    "肺部CT": "CT",
    "TBC涂片": "TBC",
  },
  inplace=True,
)

s["Gender"] = s.apply(lambda row: 1 if row["Gender"] == "女" else 0, axis=1)
s["Cough"] = s.apply(lambda row: 1 if row["Cough"] == "有" else 0, axis=1)
s["Fever"] = s.apply(lambda row: 1 if row["Fever"] == "有" else 0, axis=1)
s["Hemoptysis"] = s.apply(lambda row: 1 if row["Hemoptysis"] == "有" else 0, axis=1)
s["CT"] = s.apply(lambda row: 1 if row["CT"] == "CT-阳性" else 0, axis=1)
s["IGRAs"] = s.apply(lambda row: 1 if row["IGRAs"] == "阳性" else 0, axis=1)
s["TBC"] = s.apply(lambda row: 1 if row["TBC"] == "阳性" else 0, axis=1)
s["TB-DNA"] = s.apply(lambda row: 1 if row["TB-DNA"] == "阳性" else 0, axis=1)

s.to_csv('./data/d-20231016.csv', index_label="id")
print(s_aptb.shape)
print(s_cdptb.shape)
print(s_itb.shape)
print(s_n_tpd.shape)