19. Machine Learning with the Experts: School Budgets

- 本章的競賽題：https://www.drivendata.org/competitions/46/box-plots-for-education-reboot/page/85/
- 本章的 Jupiter notebook: https://github.com/datacamp/course-resources-ml-with-experts-budgets/blob/master/notebooks/1.0-full-model.ipynb
- EDA: Exploratory Data Analysis
- df.info(), df.head(), df.tail(), df.describe()
- df.dtypes 看各個欄位的型態
- df.dtypes.value_counts() 計算各個形態的數目
- plt.hist(df['FTE'].dropna()) 畫圖時記得不可以有 NaN，所以要用 .dropna() 把 NaN 拿掉

- 機器學習只能用在數值資料上，無法用在字串上，所以要把字串資料轉換成數值資料。
在 Pandas 中如果字串資料是分門別類的，可以先將它轉成 category 型態：df.欄位.astype('category')
- .astype() 只適用於 Pandas Series
然後再轉成數值資料：pd.get_dummies(df[['欄位']], prefix_sep='_') binary indicator
- categorize_label = lambda x: x.astype('category')
df[[欄位列表]].apply(categorize_label, axis=0) 可以對指定的欄位做轉成 category 型態的動作
- df[[欄位列表]].dtypes 檢查轉換的結果是否正確
- pd.Series.nunique 計算在一個 Series 中每一個唯一的值出現的次數
- 例如：num_unique_labels = df[[欄位列表]].apply(pd.Series.nunique)
- 畫出來看有多少個唯一的值 num_unique_labels.plot(kind='bar')
- log loss 就是 loss function，用來測量誤差，希望誤差越小越好
logloss = -\frac{1}{N} \sum_{I=1}^{N} (y_{i}\log(p_{I}) + (1 - y_{I})log(1 - p_{I}))
y 是真實的值，p 是機率
less confident 的 log loss 會比 confident and wrong 的 log loss 還要好
- import numpy as np
def compute_log_loss(expected, actual, eps=1e-14):
    predict = np.clip(predicted, eps, 1-eps) # 限制在 0 和 1 之間，但是因為要取 log 所以用一個很小的數 eps 讓 predicted 不是剛剛好等於 0 或 1 而是非常靠近 0 或 1 而已
-     loss = -1 * np.mean(actual * np.log(predicted)
-                   + (1 - actual) * np.log(1 - predicted))
-     return loss
- log loss 越小越好
logloss(actual) < logloss(correct & confident) < logloss(correct & not confident) < logloss(wrong & not confident) < logloss(wrong & confident)

- 首先先由 simple model 開始做，然後再漸漸改進成 complex model
simple model 可以快速知道結果，就知道題目難不難
complex model 容易做錯

- 由於資料的特性，有些時候 train_test_split() 並不適用。像是如果有些資料出現的頻率很低，那在分 train 和 test 時就可能只在 train 或 test 其中之一才有，另一個沒有
改用這個 multilabel_train_test_split() https://github.com/drivendataorg/box-plots-sklearn/blob/master/src/data/multilabel.py
- StratifiedShuffleSplit 只適用於單一 target 變數的情況
- OneVsRestClassifier 把每個欄位要預測的 y 都當作獨立的，分別對每個欄位做預測
- from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
clf = OneVsRestClassifier(LogisticRegression()) 本質上仍是做 Logistic regression，只是用了 OneVsRestClassifier 會對每個欄位分別做 logistic regression
- .predict_proba() 會預測機率
- .predict() 會預測結果，可是有可能造成 logloss 是 confident & wrong 的情況
- prediction_df = pd.DataFrame(columns=pd.get_dummies(df[LABELS], prefix_sep='__').columns, index=holdout.index, data=predictions)
- prefix_sep='__' 是會把 dummy 的欄位用改成 欄位__數字
prediction_df.to_csv('predictions.csv')
score = score_submission(pred_path='predictions.csv')
- NLP: Natural Language Processing
資料來源可以是任何有文字的東西，例如純文字文本，文件，演講稿等等
- Tokenization 是把文字拆成一個一個字串的動作，把每個字串存在列表中
例如：'Natural Language Processing' --> ['Natural', 'Language', 'Processing']
- 'Natural', 'Language', 'Processing' 這三個字串各自代表一個 token
- 一般來說 token 是由空白符號來分隔，但是也可以指定用其他的分隔符號
- Bag of words: 想像成一個裝有許多單字的大袋子，我們要計算某個單字在這個大袋子中出現的次數。由於已經用 NLP 拆成單字了，所以已經失去了單字在句子中的順序的資訊。
bag-of-words representation 把一長串 string 依照 token 拆解成一個個字串
- n-gram: 由 n 個 token 組成的東西
例如：每個單字就是一個 token，所以就形成 1-gram，但是兩個單字可以形成 2-gram，n 個單字組成的就變成了 n-gram
- CountVectorizer() 會把字串變成 tokens，建立 vocabulary，然後計算每個 token 在 vocabulary 中出現的次數
- CountVectorizer expects a vector, you'll need to use the preloaded function,
需要把每一個列都當成單一的字串，才能餵給 CountVectorizer() 使用
- 例如：
from sklearn.feature_extraction.text import CountVectorizer 用來拆解成字串
- TOKENS_BASIC = '\\S+(?=\\s+)' 用正規表示式指明分隔符號，這邊用的是任何的非空白符號
- df.Program_Description.fillna('', inplace=True) 不可以有空的儲存格，所以要先 .fillna()，可以加參數 inplace=True 就直接修改原本的 df (不是產生新的)
- vec_basic = CountVectorizer(token_pattern=TOKENS_BASIC) 指明用什麼分隔符號來隔開 token
- vec_basic.fit(df.Program_Description) 取出 Program_Description 欄位的 token
- msg = 'There are {} tokens in Program_Description if tokens are any non-whitespace'
- print(msg.format(len(vec_basic.get_feature_names()))) 用 vec_basic.get_feature_names() 傳回一個包含全部 token 的列表
- 例如：
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)' 用字母與數字當分隔符號
- # Fill missing values in df.Position_Extra
df.Position_Extra.fillna('', inplace=True) 填補 NaN
- # Instantiate the CountVectorizer: vec_alphanumeric
vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)
# Fit to the data
vec_alphanumeric.fit(df.Position_Extra) 取出 Position_Extra 欄位的 token
- # Print the number of tokens and first 15 tokens
msg = "There are {} tokens in Position_Extra if we split on non-alpha numeric"
print(msg.format(len(vec_alphanumeric.get_feature_names())))
print(vec_alphanumeric.get_feature_names()[:15]) 列出前 15 個 token

- 要把 DataFrame 中的每一列都轉換成單一字串，然後才能餵給 CountVectorizer() 使用
再用 .fit_transform() 把 vectorize 物件來轉換成 bag-of-words
- # Define combine_text_columns()
def combine_text_columns(data_frame, to_drop=NUMERIC_COLUMNS + LABELS):
    """ converts all text in each row of data_frame to single vector """
    # Drop non-text columns that are in the df
    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop, axis=1) 首先先要用 .drop() 把數值的欄位移除掉
-     # Replace nans with blanks
    text_data.fillna("", inplace=True) 然後填補 NaN
-     # Join all text items in a row that have a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1) 最後沿著列的方向，把每個欄位的結果結合成一個字串
- 例如：
# Import the CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# Create the basic token pattern
TOKENS_BASIC = '\\S+(?=\\s+)' 用非空白字元當作分隔符號
- # Create the alphanumeric token pattern
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)' 用字母與數字當作分隔符號
- # Instantiate basic CountVectorizer: vec_basic
vec_basic = CountVectorizer(token_pattern=TOKENS_BASIC) 初始化 CountVectorizer 物件
- # Instantiate alphanumeric CountVectorizer: vec_alphanumeric
vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC) 初始化 CountVectorizer 物件
- # Create the text vector
text_vector = combine_text_columns(df) 把 df 裡面的每一列變成是單一字串，這樣 CountVectorizer 才吃
- # Fit and transform vec_basic
vec_basic.fit_transform(text_vector) 取出 token 變成 vectorize 物件再轉換成 bag-of-words
- # Print number of tokens of vec_basic
print("There are {} tokens in the dataset".format(len(vec_basic.get_feature_names()))) 計算列表中有幾個 token
- # Fit and transform vec_alphanumeric
vec_alphanumeric.fit_transform(text_vector) 取出 token 變成 vectorize 物件再轉換成 bag-of-words
- # Print number of tokens of vec_alphanumeric
print("There are {} alpha-numeric tokens in the dataset".format(len(vec_alphanumeric.get_feature_names()))) 計算列表中有幾個 token
- Pipeline 可以把許多步驟結合成一個一系列的步驟，前者的輸出會變成後者的輸入
每個步驟是由一個有兩個元素的 tuple 組成的，(名稱, transform 物件)，物件是用來實作 .fit() 或 .transform()
- 步驟可以是另外一個 pipleline
例如：
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
pl = Pipeline([('clf', OneVsRestClassifier(LogisticRegression()))])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric']], pd.get_dummies(sample_df['label']), random_state=2)
- pl.fit(X_train, y_train)
accuracy = pl.score(X_test, y_test)
from sklearn.preprocessing import Imputer 當欄位有空的時候，imputer 會補空值，預設是用 mean
- X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing']], pd.get_dummies(sample_df['label']), random_state=2)
- pl = Pipeline([('imp', Imputer()), ('clf', OneVsRestClassifier(LogisticRegression()))]) 在 pipeline 中加入 imputer
- pipeline.fit(X_train, y_train)
accuracy = pl.score(X_test, y_test)
例如：
# Import Pipeline
from sklearn.pipeline import Pipeline
# Import other necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
# Split and select numeric data only, no nans
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric']], pd.get_dummies(sample_df['label']), random_state=22)
- # Instantiate Pipeline object: pl
pl = Pipeline([('clf', OneVsRestClassifier(LogisticRegression()))])
- # Fit the pipeline to the training data
pl.fit(X_train, y_train)
# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
例如：
# Import the Imputer object
from sklearn.preprocessing import Imputer
# Create training and test sets using only numeric data
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing']], pd.get_dummies(sample_df['label']), random_state=456)
- # Insantiate Pipeline object: pl
pl = Pipeline([('imp', Imputer()), ('clf', OneVsRestClassifier(LogisticRegression()))])
- # Fit the pipeline to the training data
pl.fit(X_train, y_train)
# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - all numeric, incl nans: ", accuracy)

- 當特徵是文字的型態時，pipleline 中用的就是 CountVectorizer
例如：
from sklearn.feature_extraction.text import CountVectorizer
X_train, X_test, y_train, y_test = train_test_split(sample_df['text'], pd.get_dummies(sample_df['label']), random_state=2)
- pl = Pipeline([('vec', CountVectorizer()), ('clf', OneVsRestClassifier(LogisticRegression()))])
- pl.fit(X_train, y_train)
accuracy = pl.score(X_test, y_test)
print('accuracy on sample data: ', accuracy)

- 當特徵有文字也有數值的型態的時候，pipleline 就沒辦法混合兩者來使用了，這時候要先使用 FunctionTransformer() 和 FeatureUnion()
- Any step in the pipeline must be an object that implements the fit and transform methods.
The FunctionTransformer creates an object with these methods out of any Python function that you pass to it.
The Imputer() imputation transformer from scikit-learn to fill in missing values in your sample data.
from sklearn.pipeline import FeatureUnion These tools will allow you to streamline all preprocessing steps for your model, even when multiple datatypes are involved.
- FunctionTransformer() 把 python 的函數轉換成 pipeline 可以使用的物件，所以文字和數值的型態各自都要使用這個功能。寫一個函數傳回 DataFrame 中的文字欄位，寫另一個函數傳回 DataFrame 中的數值欄位。然後用這個 FunctionTransformer() 轉換成物件後，將兩者的轉換結果分別餵給兩個 pipelines
- 例如：
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing', 'text']], pd.get_dummies(sample_df['label']), random_state=2)
- from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
get_text_data = FunctionTransformer(lambda x: x['text'], validate=False) validate=False 表示不做 imputation 或是檢查 dtype
- get_numeric_data = FunctionTransformer(lambda x: x[['numeric', 'with_missing']], validate=False)
- FeatureUnion() 是用來把文字和數值形態的陣列結合再一起，這樣才能一起餵給 sklearn 做機器學習
例如：
from sklearn.pipeline import FeatureUnion
union = FeatureUnion([('numeric', numeric_pipeline), ('text', text_pipeline)])

- 最後合併兩者
例如：
numeric_pipeline = Pipeline([('selector', get_numeric_data), ('imputer', Imputer())])
- text_pipeline = Pipeline([('selector', get_text_data), ('vectorizer', CountVectorizer())])
- pl = Pipeline([('union', FeatureUnion([('numeric', numeric_pipeline), ('text', text_pipeline)])), ('clf', OneVsRestClassifier(LogisticRegression()))])

- 處理文字特徵
# Import the CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# Split out only the text data
X_train, X_test, y_train, y_test = train_test_split(sample_df['text'], pd.get_dummies(sample_df['label']), random_state=456)
- # Instantiate Pipeline object: pl
pl = Pipeline([('vec', CountVectorizer()), ('clf', OneVsRestClassifier(LogisticRegression()))])
- # Fit to the training data
pl.fit(X_train, y_train)
# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - just text data: ", accuracy)

- 使用 FunctionTransformer() 同時處理文字和數值的特徵
# Import FunctionTransformer
from sklearn.preprocessing import FunctionTransformer
# Obtain the text data: get_text_data
get_text_data = FunctionTransformer(lambda x: x['text'], validate=False)
# Obtain the numeric data: get_numeric_data
get_numeric_data = FunctionTransformer(lambda x: x[['numeric', 'with_missing']], validate=False)
# Fit and transform the text data: just_text_data
just_text_data = get_text_data.fit_transform(sample_df)
# Fit and transform the numeric data: just_numeric_data
just_numeric_data = get_numeric_data.fit_transform(sample_df)
# Print head to check results
print('Text Data')
print(just_text_data.head())
print('\nNumeric Data')
print(just_numeric_data.head())

- 使用 FeatureUnion() 來預測結果
# Import FeatureUnion
from sklearn.pipeline import FeatureUnion
# Split using ALL data in sample_df
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing', 'text']], pd.get_dummies(sample_df['label']), random_state=22)
- # Create a FeatureUnion with nested pipeline: process_and_join_features
process_and_join_features = FeatureUnion(transformer_list = [('numeric_features', Pipeline([('selector', get_numeric_data), ('imputer', Imputer())])), ('text_features', Pipeline([('selector', get_text_data), ('vectorizer', CountVectorizer())]))])
- # Instantiate nested pipeline: pl
pl = Pipeline([('union', process_and_join_features), ('clf', OneVsRestClassifier(LogisticRegression()))])
- # Fit pl to the training data
pl.fit(X_train, y_train)
# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - all data: ", accuracy)

# Import FunctionTransformer
from sklearn.preprocessing import FunctionTransformer

# Get the dummy encoding of the labels
dummy_labels = pd.get_dummies(df[LABELS])

# Get the columns that are features in the original df
NON_LABELS = [c for c in df.columns if c not in LABELS]

# Split into training and test sets
X_train, X_test, y_train, y_test = multilabel_train_test_split(df[NON_LABELS],
  dummy_labels,
  0.2,
  seed=123)

# Preprocess the text data: get_text_data
get_text_data = FunctionTransformer(combine_text_columns, validate=False)

# Preprocess the numeric data: get_numeric_data
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)

# Complete the pipeline: pl
pl = Pipeline([
  ('union', FeatureUnion(
  transformer_list = [
  ('numeric_features', Pipeline([
  ('selector', get_numeric_data),
  ('imputer', Imputer())
  ])),
  ('text_features', Pipeline([
  ('selector', get_text_data),
  ('vectorizer', CountVectorizer())
  ]))
  ]
  )),
  ('clf', OneVsRestClassifier(LogisticRegression()))
  ])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)

# Import random forest classifer
from sklearn.ensemble import RandomForestClassifier

# Edit model step in pipeline
pl = Pipeline([
  ('union', FeatureUnion(
  transformer_list = [
  ('numeric_features', Pipeline([
  ('selector', get_numeric_data),
  ('imputer', Imputer())
  ])),
  ('text_features', Pipeline([
  ('selector', get_text_data),
  ('vectorizer', CountVectorizer())
  ]))
  ]
  )),
  ('clf', RandomForestClassifier())
  ])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)

引入 unigram 和 bi-gram 可以提高預測的分數
vec = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC, ngram_range=(1, 2))
holdout = pd.read_csv('HoldoutData.csv', index_col=0)
In [4]: predictions = pl.predict_proba(holdout)
In [5]: prediction_df = pd.DataFrame(columns=pd.get_dummies(
  ...: df[LABELS]).columns, index=holdout.index,
  ...: data=predictions)
In [6]: prediction_df.to_csv('predictions.csv')
In [7]: score = score_submission(pred_path='predictions.csv')

# Import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# Add model step to pipeline: pl
pl = Pipeline([
  ('union', FeatureUnion(
  transformer_list = [
  ('numeric_features', Pipeline([
  ('selector', get_numeric_data),
  ('imputer', Imputer())
  ])),
  ('text_features', Pipeline([
  ('selector', get_text_data),
  ('vectorizer', CountVectorizer())
  ]))
  ]
  )),
  ('clf', RandomForestClassifier(n_estimators=15))
  ])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)

import numpy as np
In [5]: import pandas as pd
In [6]: df = pd.read_csv('TrainingSetSample.csv', index_col=0)
In [7]: dummy_labels = pd.get_dummies(df[LABELS])
In [8]: X_train, X_test, y_train, y_test = multilabel_train_test_split(

- ...:   df[NON_LABELS], dummy_labels,
- ...:  0.2)
- get_text_data = FunctionTransformer(combine_text_columns,
...: validate=False)
- In [11]: get_numeric_data = FunctionTransformer(lambda x:
...: x[NUMERIC_COLUMNS], validate=False)
- In [12]: pl = Pipeline([
...:
...:
...:
...:
...:
...:
...:
...:
...:
...:
...:
...:
...: ])
('union', FeatureUnion([
- ('numeric_features', Pipeline([
- ('selector', get_numeric_data),
- ('imputer', Imputer())
- ])),
('text_features', Pipeline([
- ('selector', get_text_data),
- ('vectorizer', CountVectorizer())

- ])) ])
),
('clf', OneVsRestClassifier(LogisticRegression()))

- pl.fit(X_train, y_train)

-

from sklearn.ensemble import
In [15]: pl = Pipeline([
...:
...:
...:
...:
...:
...:
...:
...:
...:
...:
...:
...:
...:
...:
('union', FeatureUnion(
- transformer_list = [
('numeric_features', Pipeline([
- ('selector', get_numeric_data),
- ('imputer', Imputer())
- ])),
('text_features', Pipeline([
- ('selector', get_text_data),
- ('vectorizer', CountVectorizer())
- ]))
] )),
   ('clf', OneVsRest(RandomForestClassifier()))

- ])

from sklearn.feature_extraction.text import CountVectorizer

# Create the text vector
text_vector = combine_text_columns(X_train)

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Instantiate the CountVectorizer: text_features
text_features = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

# Fit text_features to the text vector
text_features.fit(text_vector)

# Print the first 10 tokens
print(text_features.get_feature_names()[:10])

These have been added in order to account for the fact that you're using a reduced-size sample of the full dataset in this course. To make sure the models perform as the expert competition winner intended, we have to apply a dimensionality reduction technique, which is what the dim_red step does, and we have to scale the features to lie between -1 and 1, which is what the scale step does.

The dim_red step uses a scikit-learn function called SelectKBest(), applying something called the chi-squared test to select the K "best" features. The scale step uses a scikit-learn function called MaxAbsScaler() in order to squash the relevant features into the interval -1 to 1.

# Import pipeline
from sklearn.pipeline import Pipeline

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Import other preprocessing modules
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import chi2, SelectKBest

# Select 300 best features
chi_k = 300

# Import functional utilities
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.pipeline import FeatureUnion

# Perform preprocessing
get_text_data = FunctionTransformer(combine_text_columns, validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Instantiate pipeline: pl
pl = Pipeline([
  ('union', FeatureUnion(
  transformer_list = [
  ('numeric_features', Pipeline([
  ('selector', get_numeric_data),
  ('imputer', Imputer())
  ])),
  ('text_features', Pipeline([
  ('selector', get_text_data),
  ('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
  ngram_range=(1, 2))),
  ('dim_red', SelectKBest(chi2, chi_k))
  ]))
  ]
  )),
  ('scale', MaxAbsScaler()),
  ('clf', OneVsRestClassifier(LogisticRegression()))
  ])

考慮 interaction term 時，
Interaction terms are a statistical tool that lets your model express what happens if two features appear together in the same row.
from sklearn.preprocessing import PolynomialFeatures
interaction = PolynomialFeatures(degree=2,
  ...: interaction_only=True,
  ...: include_bias=False) bias 指的是是否允許有y 截距
有 interaction term 時，數目會指數性增加， vectorizer 會把結果存成 sparse matrix 但是 PolynomialFeatures 不支援 sparse matrix 要改用 SparseInteractions
SparseInteractions does the same thing as PolynomialFeatures, but it uses sparse matrices to do so. You can get the code for SparseInteractions at this GitHub Gist. https://github.com/drivendataorg/box-plots-sklearn/blob/master/src/features/SparseInteractions.py
SparseInteractions(degree=2).fit_transform(x).toarray()

# Instantiate pipeline: pl
pl = Pipeline([
  ('union', FeatureUnion(
  transformer_list = [
  ('numeric_features', Pipeline([
  ('selector', get_numeric_data),
  ('imputer', Imputer())
  ])),
  ('text_features', Pipeline([
  ('selector', get_text_data),
  ('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
  ngram_range=(1, 2))),
  ('dim_red', SelectKBest(chi2, chi_k))
  ]))
  ]
  )),
  ('int', SparseInteractions(degree=2)),
  ('scale', MaxAbsScaler()),
  ('clf', OneVsRestClassifier(LogisticRegression()))
  ])

HashingVectorizer acts just like CountVectorizer in that it can accept token_pattern and ngram_range parameters. The important difference is that it creates hash values from the text, so that we get all the computational advantages of hashing!
# Import HashingVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

# Get text data: text_data
text_data = combine_text_columns(X_train)

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Instantiate the HashingVectorizer: hashing_vec
hashing_vec = HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

# Fit and transform the Hashing Vectorizer
hashed_text = hashing_vec.fit_transform(text_data)

# Create DataFrame and print the head
hashed_df = pd.DataFrame(hashed_text.data)
print(hashed_df.head())

# Import the hashing vectorizer
from sklearn.feature_extraction.text import HashingVectorizer

# Instantiate the winning model pipeline: pl
pl = Pipeline([
  ('union', FeatureUnion(
  transformer_list = [
  ('numeric_features', Pipeline([
  ('selector', get_numeric_data),
  ('imputer', Imputer())
  ])),
  ('text_features', Pipeline([
  ('selector', get_text_data),
  ('vectorizer', HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
  non_negative=True, norm=None, binary=False,
  ngram_range=(1, 2))),
  ('dim_red', SelectKBest(chi2, chi_k))
  ]))
  ]
  )),
  ('int', SparseInteractions(degree=2)),
  ('scale', MaxAbsScaler()),
  ('clf', OneVsRestClassifier(LogisticRegression()))
  ])
