# 8. pandas Foundations

## 小技巧
* exploratory data analysis (EDA)
* NaN 表示 Not-a-Number
* 用 Numpy 取 log，以 10 為底: `np_vals_log10 = np.log10(np_vals)`
  *  也可以把 df 當參數直接傳給 numpy methods 例如：`df_log10 = np.log10(df)` 結果是一個 DataFrame
* `eval(x)`: The `eval()` function evaluates the specified expression, if the expression is a legal Python statement, it will be executed.

## Pandas Dataframe 的基本操作
* pandas DataFrame 是一個 2D array 
  * 每一個欄位是 `pandas Series`，是一個 1D array
* 建立 Dataframe
  * data 是一個字典，keys 是欄位的名字 values 是欄位的值
  * 可以用列表來產生字典

```python
zipped = list(zip(list_keys, list_values)) # keys 和 values 分別都是 list 的時候，先產生 zip 物件的列表
data = dict(zipped) # 再把 list 轉成 dict

# 方法一: 用字典建立 DataFrame
df = pd.DataFrame(data) # 再餵給 pd.DataFrame

# 方法二: 用 list of list 建立 DataFrame
df = pd.DataFrame([
  [values of row 1],
  [values of row 2],
  [values of row 3]
], columns=[list of column names])
```

* 讀 CSV 檔到 Dataframe

```python
# 讀入 file.csv 成 DataFrame
df = pd.read_csv('file.csv')
# 讀入時可以有許多參數:
# header=n 指明用第 n 列當 header，n 從 0 開始算起，若 header=None 指明沒有 header
# names=[new_labels] 就是改成用自己指定的 header，new_labels 是一個包含 header 的列表，指明每一個欄位的名字
# na_values='-1' 會把欄位中所有 -1 替換成 NaN 也可以用 na_values={欄位名:[要替換成 NaN 的列表]}
# parse_dates=True 表示用 datatime64 格式
# parse_dates=[[0, 1, 2]] 把第 0, 1, 2 欄組成 datatime64 的日期，注意是用 list of list
# delimiter=' ' 用空白當分隔符號
# comment='#' 是說以 # 開頭的都當註解不要讀入
# index_col='A 欄位' 用 A 欄位當成 row index 那一欄
```

* 顯示 Dataframe 的相關資訊

```python
df.head(n) # 顯示頭 n 列，預設 n=5
df.tail(n) # 顯示末 n 列，
df.info() # 顯示 df 的資訊
df.shape # 顯示 df 的維度
df.index # 顯示 row index
df.columns # 顯示 columns 的名字 (型態是 index)
```

* 為欄位或是列命名

```python
# 命名列: 指定 row index
df.index=[row index 的列表]

# 命名欄位: 指定 column 的名字
df.columns=[欄位名字的列表]

# 命名欄位: 指定 row index 所在的那一個欄位的名字
df.index.name = '列的 index 那一欄的名字'
```

* `df.column_name.values` 或 `df['column_name'].values` 型態是 `numpy.ndarray`
* `df.values` 和 `pd_series.values` 也是 `numpy.ndarray` 型態
* 把 dataframe 存檔

```python
# 寫入 csv 檔
df.to_csv('file.csv', index=False) # index=False 是說不要把 index 那一個欄位寫入 csv
# 加上 sep='\t' 變成用 tab 分隔 (預設是用 , 分隔)

# 寫入 Excel 檔
df.to_excel('file.xlsx', index=False) 
```

* 計算欄或列的平均值

```python
df.mean() # 計算每個欄位的平均值
df.mean(axis='columns') # 計算每個列的平均值
# 一個列有很多欄位，axis='columns' 就是用該列全部欄位來算此列的平均值
```

* 算 quantile: 
  * quartile, quantile, percentile 的意思都類似，只是不同稱呼，下表列出互相對應的

  	|quartile|quantily|percentile|
  	|:---:|:---:|:---:|
  	|0|0|0|
  	|1|0.25|25|
  	|2|0.5|50|
  	|3|0.75|75|
  	|4|1|100|
   
  * `df.quantile(q)`: q 可以是介於 0~1 的數值或是列表
    *  Example：`df.quantile([0.05, 0.95])`, 表示算 5% 和 95% 的 quartile
    *  `df.quantile(0.5)` 等同於 median
* `df_new = df[df["欄位名"] == 數值]` 會選出 `"欄位名"` 的值是 `數值` 的所有列，放到新的 DataFrame
* the concept of 'method chaining': `df.method1().method2().method3()`
* 對 str 型態的欄位做操作

```python
df.columns.str.strip() # 欄位名字有空格就拿掉空格
df['欄位名'].str.upper() # 把該欄位所有儲存格都改大寫
df['欄位名'].str.contains('字串') # 若是欄位中的儲存格找到"字串"則該儲存格傳回 True

pd.to_numeric(pd_series, errors='coerce') # 把字串型態的數字改成數值型態的數字
# errors='coerce' 使得非數字的字串在轉成數值時變成 NaN 而不會出錯

# Example:
df_clean['dry_bulb_faren'] = pd.to_numeric(df_clean['dry_bulb_faren'], errors='coerce')
# 一開始的 df_clean['dry_bulb_faren'] 是一個 pd.Series，每個元素都是字串型態
# 用 pd.to_numeric() 就把這個 series 的所有元素都改成數值型態
```

* 刪除某些欄位: `df_dropped = df.drop(list_to_drop, axis='columns')`
* 把某個欄位設為 index: `daily_temp_climate = daily_climate.reset_index()['Temperature']`
* 可以把某 str 型態的欄位改成 datetime64 然後再把該欄位設為 index

```python
df.Date = pd.to_datetime(df.Date)
df.set_index('Date', inplace=True)
```

## 畫圖
* 畫圖可以用 Numpy array, pandas Series, 和 pandas DataFrame:
  * 用 Numpy darray 畫圖:
    * `plt.plot(numpy.ndarray)`
    * 用 Numpy 畫圖時要先把 `df['欄位'].values` 變成 numpy.ndarray, 再用 `plt.plot(numpy ndarray)` 畫圖
  * 用 Pandas Series 畫圖:
    * `plt.plot(pandas Series)
    *  `pandas_Series.plot()`
    *  `df['欄位'].plot()`
  * 用 Pandas DataFrame 畫圖:
    * `plt.plot(df)` 
    * `df.plot()`
    * 上面兩種方式會把全部的欄位畫在同張圖上 
    * `df.plot(subplots=True)` 把不同欄位畫在不同的 subplot 上，不需要自己切割子圖
* 一些畫圖的指令

```python
# y 軸改成 log scale
plt.yscale('log')

# 對圖形做各種設定
df['欄位'].plot(color='b', style='.-', legend=True) # 顏色用 b (藍色)，線用 .-，有 legend

# 設定軸的上下界線
plt.axis((xmin, xmax, ymin, ymax))

# 存圖檔，支援 png, jpg, pdf
plt.savefig('figure.png/jpg/pdf')

# 畫 histogram
# 這三種方式雖然都是畫 histogram，但是畫出來的結果有點差別，scatter, box 等圖也類似用法
# 畫 hist 時 bins=數目可指定幾個 bin，range=(low, high) 指定範圍
# 畫 hist 的 PDF 時要加上 normed=True 
# 畫 CDF (cumulative density functions) 要加上 normed=True, cumulative=True
df.plot(kind='hist', x='欄位名' y='欄位名')
df.plot.hist()
df.hist()

# 用 kind 來指定畫出各種圖
# 不指定 kind 預設就是畫 line plot
# 畫 scatter plot 時用 s=[點的大小] 來指定每個點的大小
# subplots=True 可以分成兩個圖，可以在畫 box plot 時使用
df.plot(kind='line/scatter/box/hist/area', x='欄位名', y='欄位名')

# 有子圖的情況
# 有用 subplots() 時，要用 ax=axes[n] 來指明畫在哪一張 subplot 上
fig, axes = plt.subplots(nrows=2, ncols=1) # 自己切割子圖

# 畫 hist ，因為 normed=True 會畫 PDF，ax=axes[0] 指明畫在第一列，有 30 個 bins，範圍是 0~3
df.fraction.plot(ax=axes[0], kind='hist', normed=True, bins=30, range=(0,.3))

# 畫 hist，因為 normed=True, cumulate=True 所以是畫 CDF，ax=axes[1] 指明畫在第二列，有 30 個 bins，範圍是 0~3
df.fraction.plot(ax=axes[1], kind='hist', normed=True, cumulative=True, bins=30, range=(0,.3))

# 選出 pclass=1 的所有列，用 fare 欄位畫 box 圖，ax=axes[0] 指明畫在第一列
titanic.loc[titanic['pclass'] == 1].plot(ax=axes[0], y='fare', kind='box')
```

## 日期時間
* 日期時間格式 `yyyy-mm-dd hh:mm:ss`
* 格式轉換
  * 用 `pd.to_datetime()` 把 str 型態的日期時間轉成 datetime64 型態
    * 轉換成 datetime64 型態的日期時間可以設成 index 給時間序列使用 

  ```python
  my_datetimes = pd.to_datetime(date_list, format='%Y-%m-%d %H:%M')
  # date_list 的元素是 str 的型態
  # my_datetimes 是 pandas.core.indexes.datetimes.DatetimeIndex 型態，每個元素是 datetime64 的型態 
   
  time_series = pd.Series(temperature_list, index=my_datetimes) # my_datetimes 現在可以設成 index 給時間序列使用
  ```

* Resample:
  * 日期時間可以 resample，分成 down-sampling 和 up-sampling
  * down-sampling: 是把日變週，週變月，月變年等等
  * up-sampling 是把年變月，月變週，週變日等等

  ```python
  series.resample(頻率).統計函數()
  ```

  * Example

  ```python
  df = df['Temperature'].resample('6h').mean() # 算六小時的平均
  df = df['Temperature'].resample('D').count() # 算每日的數目
  
  ```

* 設定時區: `times_tz_central = times_tz_none.dt.tz_localize('US/Central')`
* 轉換時區: `times_tz_pacific = times_tz_central.dt.tz_convert('US/Pacific')`
* Reindex:
  * 時間序列的 index 可以拿別的時間序列的 index 來重設，但是當兩個時間序列的長度不同的時候，需要對時間序列做一些操作

```python
ts4 = ts2.reindex(ts1.index, method='ffill') 把 ts2 的 index 用 ts1 的 index 取代
# method='ffill' (forward fill) 也可以用 method='bfill' (backward fill)
# 因為 ts1 和 ts2 的長度不同 (len(ts1) > len(ts2))，所以需要用 ffill 或是 bfill
# 把 ts2 所缺少的值填上去

ts2_interp = ts2.reindex(ts1.index).interpolate(how='linear')
# 這邊是用內插法填上缺少的值
```

* **Rolling means** (or **moving averages**) are generally used to smooth out short-term fluctuations in time series data and highlight long-term trends.
  * `hourly_data.rolling(window=24).mean()` would compute new values for each hourly point, based on a 24-hour window stretching out behind each point. The frequency of the output data is the same: it is still hourly.