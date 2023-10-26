import pandas as pd

s_atb = pd.read_csv('./data/结核数据20230926/ATB-Table 1.csv')
s_cdtb = pd.read_csv('./data/结核数据20230926/CDTB-Table 1.csv')
s_ntb = pd.read_csv('./data/结核数据20230926/NTB-Table 1.csv')
s_ntm = pd.read_csv('./data/结核数据20230926/NTM-Table 1.csv')

s_atb['label'] = 'atb'
s_cdtb['label'] = 'cdtb'
s_ntb['label'] = 'ntb'
s_ntm['label'] = 'ntm'

s = pd.concat([s_atb, s_cdtb, s_ntb, s_ntm], axis=0)

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

# s.to_csv('./data/d-20230926.csv', index_label="id")
print(s_atb.shape)
print(s_cdtb.shape)
print(s_ntb.shape)
print(s_ntm.shape)