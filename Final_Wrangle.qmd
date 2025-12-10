---
title: "Untitled"
format: html
---

Importing all the neccessary packages to run the below code
```{python}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
```

Found this dataset on Kaggle. Its called Kidney Function Health dataset. I am going to fins out if diabetes and any of these variables are correlated. 
```{python}
kidney = pd.read_csv('C:/Users/tyube/Downloads/kidney_dataset.csv')
```

Doing this code to split the variables up into numerical columns which is what I want to work with. 
```{python}
numeric_cols = kidney.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "Diabetes"]
numeric_cols
```

Getting the means for all the numerical values to see if any stick out across the diagnosed and undiagnosed group. 
```{python}
group_means = kidney.groupby("Diabetes")[numeric_cols].mean().T
group_means
```

I visualized this in a heatmap becasue I like the colors and it is a good way to visualize them next to eachother. 
```{python}
plt.figure(figsize=(8, 6))
sns.heatmap(group_means, annot=True, fmt=".2f", cmap="viridis")
plt.title("Mean Feature Values by Diabetes Status")
plt.tight_layout()
plt.show()

```

Wanted to see if age was a significant fatcor in this so i compared the age of people who were diagnosed and those who were not. 
```{python}
plt.figure(figsize=(6, 4))
sns.boxplot(x="Diabetes", y="Age", data=kidney)
plt.title("Age vs Diabetes")
plt.tight_layout()
plt.show()
```

I brought in a bit of the machine learning class and wanetd to run a model to see how well certain variables predicted diabetes. 
```{python}
y = kidney["Diabetes"]
X = kidney.drop(columns=["Diabetes"])
```


```{python}
X = pd.get_dummies(X, drop_first=True)

X.head()
```

Setting up my train and test data by setting the train percentage. 
```{python}
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

Scaled it to cut down train times and normalize my data comparisons. 
```{python}
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Running the log regressions here to get the predictions. 
```{python}
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
```

Usig .predict to get my predicted coefficients for the varibles. 
```{python}
y_pred = log_reg.predict(X_test_scaled)
y_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
y_test
y_pred
```

This is to see how well my model performed. 
```{python}
auc = roc_auc_score(y_test, y_proba)
auc
```

Actually predicted very well

Sorting them from most to least to see which are the ten most important and the ten least important. 
```{python}
coef = pd.Series(log_reg.coef_[0], index=X.columns)
coef_sorted = coef.sort_values(ascending=False)
```


```{python}
coef_sorted.head(10)
```


```{python}
coef_sorted.tail(10)
```

Plotting the variables with the most influence on diabetes diagnosis. 
```{python}
plt.figure(figsize=(8, 6))
coef_sorted.head(10).plot(kind="bar")
plt.title("Top Positive Predictors of Diabetes")
plt.ylabel("Log-Odds Coefficient")
plt.tight_layout()
plt.show()
```


Plotting the top predictors that fight against diabetes. 
```{python}
plt.figure(figsize=(8, 6))
coef_sorted.tail(10).plot(kind="bar")
plt.title("Top Negative Predictors of Diabetes")
plt.ylabel("Log-Odds Coefficient")
plt.tight_layout()
plt.show()
```