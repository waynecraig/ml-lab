import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap
import umap.plot
import matplotlib.pyplot as plt

s = pd.read_csv("./data/d-20230922.csv")

y = s["label"]
X = s.drop(columns=["label"])

X_std = StandardScaler().fit_transform(X)

color_key = {
  "ntb": "blue",
  "ntm": "yellow",
  "tb": "red",
}

mapper = umap.UMAP().fit(X_std)
umap.plot.points(mapper, labels=y, theme="fire", color_key=color_key)
plt.savefig("./data/umap.png")
plt.close()