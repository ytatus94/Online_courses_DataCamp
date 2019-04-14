# 2. Intermediate Python for Data Science

## Matplotlib

```python
import matplotlib.pyplot as plt
```

* 點與線的圖

```python
plt.plot([x values], [y values]) # 畫圖，每個點用線連起來
plt.plot([list]) # 如果只有一個 list 則用 list 的 index 當 x 軸，用 list 的值當 y 軸

plt.plot(np_array) # 除了可以用 list 當參數以外，也可以用 Numpy array，和 Pandas Series 當參數。
plt.plot(pd_Series)
```

* 散射圖

```python
plt.scatter([x values], [y values], s=[點的大小], c=[點的顏色], alpha=0 到 1 之間的透明度)
```

* histogram

```python
plt.hist([values], bins=數字)
```

* 座標軸等圖表裝飾

```python
# 指定座標軸名稱和圖的名稱
plt.xlabel('x title')
plt.ylabel('y title')
plt.title('canvas title')

# 若有指定 tick labels 則座標軸上會顯示指定的 labels 當成標示，不然就用 values 當成標示
plt.xticks([tick values], [tick labels])
plt.yticks([tick values])

# 加入文字
plt.text(x座標, y座標, '字串')

# 改成 log 座標
plt.xscale('log')

# 顯示格線
plt.grid(True)

# 把目前視窗上的圖片清除掉
plt.clf()
```

---

* `dict.keys()` 列出字典內的所有 key
* `A in dict` 去找看看 A 是否是 dict 的 key
* `del(某東西)` 刪除某東西

## Pandas

```python
import pandas as pd
```

* `df.index=[新的 row labels]`, `df.index.name='row label 那個欄位的名字'`
  * 例如：`cars.index = ['Honda', 'Toyota', 'Ford']` 指定 row labels 為 Honda, Toyota, 和 Ford
* `df.columns=[新的 column labels]`
   * 沒指定的話預設就用 0, 1, 2, 3...當 labels
* 建立 Pandas DataFrame
```python
pd.DataFrame(字典)
``` 
  * 字典的 key 變成欄位名，value 變成表格的值。
    * 例如：`cars = pd.DataFrame(my_dict)` 其中 `my_dict={'a':[list_a], 'b':[list_b]}`
  * 也可以用 list 來建立 DataFrame。
* 從 csv 中讀入 DataFrame: `df = pd.read_csv('file.csv', index_col=0)`
  * `index_col=0` 指明用 column=0 當 row labels，預設會把 csv 檔案中的第一列讀成 column labels

### 存取 DataFrame：

```python
df['欄位名'] # 單括號傳回的是 Pandas Series，Pandas Series 是有 index 的一維陣列
df[['欄位名']] # 雙括號傳回的是 Pandas DataFrame
df[['欄位名1', '欄位名2']] # 存取兩個欄位，傳回 Pandas DataFrame
```

* `df[0:5]` 選出前五列，注意用 index 的話是存取列，用欄位的名字來存取欄
* `df.loc['row label']` 用 row label 來存取 row
  * `df.loc[['row labels'], ['欄位名']]` 用列與欄位的名字傳回交集元素，可以傳回多欄多列
* `df.iloc[row index]` 用 row index 來存取 row，`iloc()` 的 i 表示 index
  * `df.iloc[3, 0]` 用列與欄位的 index 傳回交集元素，傳回的是元素的值
  * `df.iloc[:, [1, 2]]` 全部的列與第二第三欄的交集
* 還有 `ix()`, `iat()`, `at()` 可以用

## 迴圈
* `for index, a in enumerate(列表):`
  * `enumerate(列表)` 會把列表的值一個個取出，並加上 index 0, 1, 2, 3... 做成 tuple。
  * index 也可以用 `enumerate(列表, start=n)` 來改成由 n 開始。
* 各種迴圈的方式：

  ```python
  for key, value in dict.items(): # 迴圈字典
  for x in np_array_1D: # 迴圈 np_array
  for x in np.nditer(np_array_2D): #  迴圈多維 Numpy array 要用 np.nditer()
  for lab, row in pd_dataframe.iterrows(): # 迴圈 data_frame 要用 .iterrows()，其中 for loop 傳回的 row data 是 Pandas Series
  ```
  
---

* 建立新的欄位方法一：用迴圈

  ```python
  for lab, row in df.iterrows() :
    df.loc[lab, "新欄位的名字"] = len(row["某欄位"])
  ```
  等號右邊的計算為新欄位賦予值，這邊是計算某個欄位的字元長度，放到新欄位中

* 建立新的欄位方法二：用 `apply()`
  ```python
  df["新欄位的名字"] = df["某欄位"].apply(len)
  ```
  函數在 `apply()` 內時不用加括號，這個和方法一會得到相同的結果
  
  ```python
  cars['COUNTRY'] = cars['country'].apply(str.upper)
  ```
  country 欄位的值是字串，所以要用 `str.upper` 才能改大寫

## Numpy
* Numpy array 的布林判斷：

```python
np.logical_and(np_array1, np_array2)
np.logical_or(np_array1, np_array2)
np.logical_not(np_array1, np_array2)
```

也可以用 Pandas Series 取代 np_array

* Numpy random number:

```python
np.random.seed(123) # 產生亂數種子
np.random.rand() # 產生 0 到 1 之間的隨機數
np.random.random(size=要產生幾個隨機數) # 傳回一個 Numpy array 元素個數是 size 所指定的數目
np.random.randint(下限, 上限) # 產生下限到上限之間的隨機數，不包含上限
np.random.randn() # 依照 Normal distribution 產生亂數
```

* `np.transpose(np_array)` 轉置
