# 18. Supervised Learning with scikit-learn

## 名詞解釋
#### Supervised & Unsupervised learning
* Supervised learning: Uses labeled data
  * kNN: k-Nearest Neighbors
  * Linear regression, Laso & Ridge regresssion
  * Logistic regression
* Unsupervised learning: Uses unlabeled data

#### Features & Targets
* Features = predictor variables = independent variables
  * 用來預測的變數，通常是很多個變數欄位
* Target variable = dependent variable = response variable
  * 要預測的變數，通常只是一個變數欄位


## DataFrame 的基本操作
* 拿到數據後首先要做的是 EDA 像是 `df.head()`, `df.info()`, `df.describe()`, `df.columns`
* `X = df.drop('欄位A', axis=1).values` 沿著 row (axis=1) 的方向把欄位 A 的值丟掉 (可以看成把整個表格的欄位 A 刪除)，再用 `.values` 把 dataframe 變成 numpy array
* `df.corr()`: computes the pairwise correlation between columns

## 畫圖相關
* 畫 scatter matrix

```python
plt.style.use('ggplot') # 指定圖的風格，這邊指定用 ggplot 的風格
plt.figure() # 要有這行才會畫在新的圖上，不然只會取代原先的圖
pd.scatter_matrix(df, c=y, figsize=[8,8], s=150, marker='D') # 畫某特徵 vs 其他特徵的圖
# c 是顏色，表示依照 y 來分類成不同的顏色
# s 是指定 marker size
plt.xticks([tick values], [tick labels]) # 改變 x 軸座標上的標示
plt.show() # 把圖秀出來
```

* `plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')`
* 畫 `sns.countplot()`:

```python
sns.countplot(x='畫哪個欄位', hue='用哪個欄位分類', data=df, palette='RdBu')
```

## Sklearn 的基本操作
* sklearn 的範例 dataset 的型態是 `Bunch` (和 `dict` 差不多)，可以用 `data.key` 或是 `data['key']` 來存取
* 不同的模型，使用 sklearn 的步驟是一樣的
  * estimators: Train-Test-Split/Instantiate/Fit/Predict paradigm applies to all classifiers and regressors

### kNN: k-Nearest Neighbors
* kNN 是由相鄰的數據點來判斷分類

```python
from sklearn.neighbors import kNeighborsClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)
# 習慣上 feature array 是 X, target array是 y
# test_size = 0.2 是把樣本切成 80% training 和 20% testing (預設是 test_size = 0.25)
# random_state=42 指定亂數種子
# stratify=y 訓練和測試樣本的標籤分佈與原始數據相同

knn = KNeighborsClassifier(n_neighbors = 6)
# 初始化，依照 6 個鄰居來判斷分類
# 鄰居數目越小 (more complex model) 越容易對某特別的事件較敏感，造成 overfitting
# 鄰居數目越大 (less complex model) 區分的邊界越平滑，但是數目太大時，會容易造成 underfitting
knn.fit(X_train, y_train) # 對訓練樣本做 fitting，參數只能是 Numpy array
y_pred = knn.predict(X_test) # 用測試樣本判斷 (可以用任何樣本來預測，例如用訓練樣本判斷，但是這樣沒有意義)
knn.score(X_test, y_test) # 求 model performance，用測試樣本評估預測的正確性 accuracy
```

### Linear regression
* linear regression 就是要把 loss function 最小化
  * Error function = cost function = loss function
* Residual 就是點到 linear regression 的線的垂直距離
* OLS (ordinary least squares): Minimize sum of squares of residuals
* regression 的分數就叫做 $R^2$

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 把樣本切成訓練與測試用
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
reg = LinearRegression() # 初始化
reg.fit(X_train, y_train) # 訓練模型
y_pred = reg.predict(X_test) # 預測結果
reg.score(X_test, y_test) # 看 model performance，R^2 的分數，但是這和如何切樣本會有關

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # 也可以求 RMSE: Root Mean Squared Error
```

* Cross-validation: 因為 $R^2$ 和如何把樣本切成訓練與測試有關，所以就乾脆多切幾次，然後來算每次的 $R^2$ 再求平均

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

reg = LinearRegression()
cv_scores = cross_val_score(reg, X, y, cv=5)
# 將樣本切成 5 份，一份當測試，剩下的當訓練。
# 所以有五種可能的訓練與測試樣本組合，這叫做 5-fold CV
# 計算的結果是一個 list，切越多份計算的時間會越久
np.mean(cv_scores) # 然後求平均值
```

* Regularization: 當有很多 feature 時 (large coefficients)，linear regression 容易造成 overfitting，regularization 就是用來 penalize large coefficients
  * Regularization 有分成 Ridge regression 和 Lasso regression 兩種
  * 通常不會直接用 linear regression，會先 regularization 正規化之後才用
* Lasso regression (L1 regularization):
  * loss = OLS loss + $\alpha \sum_{i}^{n} |a_{i}|$ 這種叫做 L1 regularization，多加上去的那項叫做 penalty term
  * lasso 會把比較不重要的特徵的係數 $a_{i}$ 縮成 0
* Ridge regression (L2 regularization)
  * loss = OLS loss + $\alpha \sum_{i}^{n}a_{i}^{2}$ 這種叫做 L2 regularization
  * 通常要做 regression 時，優先選擇用 Ridge regression
* 在 sklearn 中用 Lasso 或是 Ridge 時，有一個參數 alpha 要調整，alpha 可以控制模型的複雜度
  * 選擇 alpha 的值就叫做 hyperparameter tuning
  * 當 alpha = 0 時就是原本的 OLS，可能會有 overfitting 的情形  
  * 當 alpha 很大時可能會有 underfitting 的情形

```python
# Lasso regression
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.4, normalize=True)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
lasso.score(X_test, y_test)
lasso_coef = lasso.coef_ # 可以得知每個係數值，可以看到不重要的係數都被縮成 0

# Ridge regression
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.1, normalize=True)
# 也可以事後用 ridge.alpha = 0.1 的方式來指定參數
# normalize=True 使得所有的變數都是維持在相同的尺度
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge.score(X_test, y_test)
```

### Logistic regression
* 雖然 Logistic regression 名字有回歸 (regression)，但是 Logistic regression 是用在分類上而不是回歸上
* 輸出機率 p>0.5 就 label=1，p<0.5 就 label=0，0.5 是 threshold 可以改
* 有一個正規化參數 (regularization parameter) `C` 來控制正規化強度的倒數 (inverse of the regularization strength)
  * `C` 就是一個 hyperparameter，`C` 越大會 overfit，`C` 越小會 underfit
* 另一個 hyperparameter 是 penalty 值是 `['l1', 'l2']`

```python 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
confusion_matrix(y_test, y_pred)
classification_report(y_test, y_pred)
```




- numpy 的 reshape(-1, 1) 的 -1 表示由原本的列x行自動計算 reshape 後的列數，然後第二個參數表示 reshape 後要有幾行
- 



- 
- 





- 混淆矩陣
假設兩類別 A B 想要判斷的類別是 A
|            | 預測 A | 預測 B |
| 真正 A | T +      | F -      |
| 真正 B | F +      | T-       |
通常把 positive 定義為我們想要判斷的類別，True 和 False 則是把判斷結果與真實結果比較
Accuracy = (tp + tn)/(tp + tn + fp + fn) 就是 判斷成功的數目 / 全部判斷的數目
- precision = PPV (positive predicted value) = tp / (tp + fp) 判斷成功的數目 / 全部被判斷為想要的類別的數目
- recall = sensitivity = tp / (tp + fn) 判斷成功的數目 / 真正該類別的數目，其實就是 true positive rate
- f1 score = 2 * (precision * recall)/(precision + recall)

- 產生分類報表和混淆矩陣
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)) 注意參數順序要先 test sample 的 label 再 predicted label
- classification_report(y_test, y_pred) 注意參數順序要先 test sample 的 label 再 predicted label
- test sample 的 label 是真的結果，predicted label 是用 test sample 預測的結果，比較兩者
分類報表中的 support 欄位是參與計算的樣本數目
- 




- 
- 

- 

- ROC 圖
在 sklearn 中幾乎所有的 classifier 都有 .predict_proba()
- from sklearn.metrics import roc_curve
# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1] 輸出有兩個欄位，分別對應到 label=0, label=1，用 index=1 表示選出 label=1 的欄位
- # Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)，fpr: False Positive Rate，tpr: True Positive Rate
- 圖要自己畫
曲線下面積 (AUC: area under ROC curve) 越大表示模型越好
- AUC 的分數計算有兩種方式，用 roc_auc_score() 或是用 cross-validation 的方式
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
y_pred_prob = logreg.predict_proba(X_test)[:,1]
roc_auc_score(y_test, y_pred_prob)
cv_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')
- Hyperparameters: 是指那些在 fitting 前必須先選擇好的參數，是無法從學習中學到的
要找到最合適的 hyperparameter 唯一的方式就是用不同的值去做 fit 然後看哪個結果最好，這個叫做 hyperparameter tuning
- GridSearchCV 就是一種 hyperparameter tuning 把不同的 hyperparameter 值做成一個 grid 然後用 cross-validation 跑每個 grid 看哪個最好
- 例如用 kNN
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': np.arange(1, 50)} 把 Hyperparameter 做成字典
- knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X, y) 一般來說，要把資料拆成訓練跟測試，然後用訓練的來 fit
- knn_cv.best_params_ 可以看哪個參數值是最好的，得到 {'n_neighbors': 12} 表示用 12 個鄰居是最好的
- knn_cv.best_score_ 最好的參數值得到的分數，例如 0.933216168717
- 如果用其他的模型，就把 kNN 的部分換掉就好
當 hyperparameters 很多個，樣本數很大，跑 GridSearchCV 就很花時間
- RandomizedSearchCV 就不用跑每個 hyperparameter，有些 hyperparameter 的值是從機率分佈中決定
- 例如用 decision tree
- from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
                        "max_features": randint(1, 9),
                        "min_samples_leaf": randint(1, 9),
                        "criterion": ["gini", "entropy"]}
# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()
# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)
# Fit it to the data
tree_cv.fit(X, y)
# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
- print("Best score is {}".format(tree_cv.best_score_))
- Elastic net 回歸:
- regularization: a * L1 + b * L2 叫做 L1_ratio 當 L1_ratio = 1 表示 a=1, b=0，若 L1_ratio < 1 則 a < 1, b > 0
# Import necessary modules
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}
# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()
# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)
# Fit it to the training data
gm_cv.fit(X_train, y_train)
# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
- print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))
- scikit-learn 只接受數值的參數，所以非數值要先用 pd.get_dummies(df) 把非數值參數改成數值的參數，可以加上參數 drop_first=True 把第一個新欄位丟掉，因為若其他欄位的數值都是空，那就表示屬於第一個新欄位
- | feature |             | feature_A | feature_B | feature_C |
| A          |             | 1                | 0               | 0              |
| B          |     變成  | 0               | 1                | 0              |

       | C          |             | 0               | 0               | 1               |

- Scikit-learn 的 OneHotEncoder() 也可以做跟 pd.get_dummies() 相同的事情
- df.isnull().sum() 檢查是否有空值
- SVM: Support Vector Machine

- 有空值時可以補值
- from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0) 沿著欄的方向把所有 NaN 用最常出現的值取代

- 用 pipeline 可以把一連串的動作組合在一起
from sklearn.pipeline import Pipeline
steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)), ('SVM', SVC())] 把要放到 pipeline 的東西按照先後順序放入 list 中
- pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
pipeline.fit(X_train, y_train) 然後可以用 pipeline 來做學習與預測等動作
- y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

- 當有些特徵的變化量很大時，並不適合直接用來做預測，因為梯度下降的速度在不同特徵上會不一樣
要先把它標準化使得平均值是0標準差是1，然後用標準化過後的來做預測
from sklearn.preprocessing import scale
X_scaled = scale(X)

- 可以用 StandardScaler 和 pipeline 一起
- from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]
- # Create the pipeline: pipeline
pipeline = Pipeline(steps)
- SVM 有兩個 hyperparameters: C 和 gamma
C: regularization strength, gamma: kernel coefficient
