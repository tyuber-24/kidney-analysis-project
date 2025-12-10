# Untitled


Importing all the neccessary packages to run the below code

``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
```

Found this dataset on Kaggle. Its called Kidney Function Health dataset.
I am going to fins out if diabetes and any of these variables are
correlated.

``` python
kidney = pd.read_csv('C:/Users/tyube/Downloads/kidney_dataset.csv')
```

Doing this code to split the variables up into numerical columns which
is what I want to work with.

``` python
numeric_cols = kidney.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "Diabetes"]
numeric_cols
```

    ['Creatinine',
     'BUN',
     'GFR',
     'Urine_Output',
     'Hypertension',
     'Age',
     'Protein_in_Urine',
     'Water_Intake',
     'CKD_Status']

Getting the means for all the numerical values to see if any stick out
across the diagnosed and undiagnosed group.

``` python
group_means = kidney.groupby("Diabetes")[numeric_cols].mean().T
group_means
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

| Diabetes         | 0           | 1           |
|------------------|-------------|-------------|
| Creatinine       | 2.000004    | 1.979882    |
| BUN              | 30.902334   | 30.486445   |
| GFR              | 74.406342   | 70.479609   |
| Urine_Output     | 1662.118314 | 1669.548539 |
| Hypertension     | 0.379887    | 0.374150    |
| Age              | 50.511468   | 48.937538   |
| Protein_in_Urine | 543.687295  | 521.124561  |
| Water_Intake     | 2.505654    | 2.501166    |
| CKD_Status       | 0.263173    | 0.262585    |

</div>

I visualized this in a heatmap becasue I like the colors and it is a
good way to visualize them next to eachother.

``` python
plt.figure(figsize=(8, 6))
sns.heatmap(group_means, annot=True, fmt=".2f", cmap="viridis")
plt.title("Mean Feature Values by Diabetes Status")
plt.tight_layout()
plt.show()
```

![](Final_Wrangle_files/figure-commonmark/cell-6-output-1.png)

Wanted to see if age was a significant fatcor in this so i compared the
age of people who were diagnosed and those who were not.

``` python
plt.figure(figsize=(6, 4))
sns.boxplot(x="Diabetes", y="Age", data=kidney)
plt.title("Age vs Diabetes")
plt.tight_layout()
plt.show()
```

![](Final_Wrangle_files/figure-commonmark/cell-7-output-1.png)

I brought in a bit of the machine learning class and wanetd to run a
model to see how well certain variables predicted diabetes.

``` python
y = kidney["Diabetes"]
X = kidney.drop(columns=["Diabetes"])
```

``` python
X = pd.get_dummies(X, drop_first=True)

X.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | Creatinine | BUN | GFR | Urine_Output | Hypertension | Age | Protein_in_Urine | Water_Intake | CKD_Status | Medication_ARB | Medication_Diuretic |
|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 0.788803 | 8.386869 | 102.161787 | 1632.649387 | 0 | 27.682074 | 106.700203 | 1.570370 | 0 | False | False |
| 1 | 3.413970 | 53.688796 | 50.071257 | 935.540516 | 0 | 33.122208 | 410.008362 | 3.425287 | 1 | False | False |
| 2 | 0.647645 | 7.466540 | 89.451831 | 1774.553846 | 1 | 55.832284 | 123.336925 | 1.123301 | 0 | False | True |
| 3 | 0.795508 | 12.516821 | 99.872180 | 2360.602980 | 0 | 32.391900 | 116.098870 | 3.086846 | 0 | False | False |
| 4 | 0.869010 | 19.855960 | 86.110182 | 1987.750901 | 1 | 66.689515 | 55.668760 | 2.174980 | 0 | True | False |

</div>

Setting up my train and test data by setting the train percentage.

``` python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

Scaled it to cut down train times and normalize my data comparisons.

``` python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Running the log regressions here to get the predictions.

``` python
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
```

|     |                    |              |
|-----|--------------------|--------------|
|     | penalty            | 'l2'         |
|     | dual               | False        |
|     | tol                | 0.0001       |
|     | C                  | 1.0          |
|     | fit_intercept      | True         |
|     | intercept_scaling  | 1            |
|     | class_weight       | None         |
|     | random_state       | None         |
|     | solver             | 'lbfgs'      |
|     | max_iter           | 1000         |
|     | multi_class        | 'deprecated' |
|     | verbose            | 0            |
|     | warm_start         | False        |
|     | n_jobs             | None         |
|     | l1_ratio           | None         |

            </details>
        </div>
    </div></div></div></div></div><script>function copyToClipboard(text, element) {
    // Get the parameter prefix from the closest toggleable content
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const fullParamName = paramPrefix ? `${paramPrefix}${text}` : text;
&#10;    const originalStyle = element.style;
    const computedStyle = window.getComputedStyle(element);
    const originalWidth = computedStyle.width;
    const originalHTML = element.innerHTML.replace('Copied!', '');
&#10;    navigator.clipboard.writeText(fullParamName)
        .then(() => {
            element.style.width = originalWidth;
            element.style.color = 'green';
            element.innerHTML = "Copied!";
&#10;            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'red';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        });
    return false;
}
&#10;document.querySelectorAll('.fa-regular.fa-copy').forEach(function(element) {
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const paramName = element.parentElement.nextElementSibling.textContent.trim();
    const fullParamName = paramPrefix ? `${paramPrefix}${paramName}` : paramName;
&#10;    element.setAttribute('title', fullParamName);
});
</script></body>

Usig .predict to get my predicted coefficients for the varibles.

``` python
y_pred = log_reg.predict(X_test_scaled)
y_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
y_test
y_pred
```

    array([1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
           1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1,
           1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1,
           1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
           0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1,
           1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0,
           0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1,
           1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0,
           1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0,
           0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
           0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
           0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
           0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
           0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
           0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0,
           0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1,
           0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0,
           0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0,
           1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
           0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0,
           1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1,
           0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
           0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1,
           0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0,
           1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
           0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
           1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
           0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
           0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
           1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
           0, 0, 0, 1, 1, 0, 0, 1, 0, 0])

This is to see how well my model performed.

``` python
auc = roc_auc_score(y_test, y_proba)
auc
```

    0.9373398084446243

Actually predicted very well

Sorting them from most to least to see which are the ten most important
and the ten least important.

``` python
coef = pd.Series(log_reg.coef_[0], index=X.columns)
coef_sorted = coef.sort_values(ascending=False)
```

``` python
coef_sorted.head(10)
```

    Urine_Output            0.311022
    Water_Intake            0.010872
    Medication_ARB          0.000878
    Medication_Diuretic    -0.012213
    Hypertension           -0.979485
    Protein_in_Urine       -1.300772
    CKD_Status             -1.415772
    Age                    -1.533746
    BUN                    -4.166445
    Creatinine            -11.084236
    dtype: float64

``` python
coef_sorted.tail(10)
```

    Water_Intake            0.010872
    Medication_ARB          0.000878
    Medication_Diuretic    -0.012213
    Hypertension           -0.979485
    Protein_in_Urine       -1.300772
    CKD_Status             -1.415772
    Age                    -1.533746
    BUN                    -4.166445
    Creatinine            -11.084236
    GFR                   -16.788682
    dtype: float64

Plotting the variables with the most influence on diabetes diagnosis.

``` python
plt.figure(figsize=(8, 6))
coef_sorted.head(10).plot(kind="bar")
plt.title("Top Positive Predictors of Diabetes")
plt.ylabel("Log-Odds Coefficient")
plt.tight_layout()
plt.show()
```

![](Final_Wrangle_files/figure-commonmark/cell-18-output-1.png)

Plotting the top predictors that fight against diabetes.

``` python
plt.figure(figsize=(8, 6))
coef_sorted.tail(10).plot(kind="bar")
plt.title("Top Negative Predictors of Diabetes")
plt.ylabel("Log-Odds Coefficient")
plt.tight_layout()
plt.show()
```

![](Final_Wrangle_files/figure-commonmark/cell-19-output-1.png)
