import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap
import umap.plot
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

s = pd.read_csv("./data/d-20230926.csv")

y = s["label"]
X = s.drop(columns=["label"])

X = StandardScaler().fit_transform(X)

oversample = RandomOverSampler()

X, y = oversample.fit_resample(X, y)

color_key = {
  "atb": "red",
  "cdtb": "yellow",
  "ntb": "blue",
  "ntm": "green",
}

mapper = umap.UMAP().fit(X)
umap.plot.points(mapper, labels=y, theme="fire", color_key=color_key)
plt.savefig("./data/result3/umap.png")
plt.close()