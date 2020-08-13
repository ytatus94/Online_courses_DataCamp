# 8. pandas Foundations

## Pandas Dataframe 的基本操作
* pandas DataFrame 是一個 2D array 
  * 每一個欄位是 pandas Series，是一個 1D array
* 顯示 Dataframe 的相關資訊
```python
# 顯示頭 n 列，預設 n=5
df.head(n)

# 顯示末 n 列，
df.tail(n)

# 顯示 df 的資訊
df.info()

# 顯示 df 的維度
df.shape

# 顯示 row index
df.index

# 顯示 columns 的名字 (型態是 index)
df.columns
```
* 為欄位或是列命名
```python
# 命名列: 指定 row index
df.index=[row index 的列表]

# 命名欄位: 指定 column 的名字
df.columns=[欄位名字的列表]

# 命名欄位: 指定 row index 所在的那一個欄位的名字
df.index.name='列的 index 那一欄的名字'
```
* `df.column_name.values` 或 `df['column_name'].values` 型態是 numpy.ndarray
* `df.values` 和 `pd_series.values` 也是 numpy.ndarray 型態
* NaN 表示 Not-a-Number
* 建立 Dataframe
  * data 是一個字典，keys 是欄位的名字 values 是欄位的值
```python
df = pd.DataFrame(data)
```
* 讀 CSV 檔到 Dataframe
```python
# 讀入 file.csv 成 DataFrame
df = pd.read_csv('file.csv')
```

- np_vals_log10 = np.log10(np_vals)
- 也可以把 df 當參數直接傳給 numpy methods 例如：df_log10 = np.log10(df) 結果是 DataFrame
- eval(x)
-  
- 可以用列表來產生字典：
zipped = list(zip(list_keys, list_values))
data = dict(zipped)
- 
- header=n 指明用第 n 列當 header，n 從 0 開始算起，若 header=None 指明沒有 header
- names=[new_labels] 就是改成用自己指定的 header，new_labels 是一個包含 header 的列表，指明每一個欄位的名字
- na_values='-1' 會把欄位中所有 -1 替換成 NaN 也可以用 na_values={欄位名:[要替換成 NaN 的列表]}
- parse_dates=[[0, 1, 2]] 把第 0, 1, 2 欄組成 datatime64 的日期，注意是用 list of list
- parse_dates=True 表示用 datatime64 格式
- delimiter=' ' 用空白當分隔符號
- comment='#' 是說以 # 開頭的都當註解不要讀入
- index_col='A 欄位' 用 A 欄位當成 row index 那一欄
- df.to_csv('file.csv', index=False) 寫入一個新的 csv 檔，index=False 是說不要把 index 那一個欄位寫入 csv，加上 sep='\t' 變成用 tab 分隔
- df.to_excel('file.xlsx', index=False) 寫入一個 Excel 檔

- 畫圖可以用 Numpy array, pandas Series, 和 pandas DataFrame 當作參數：plt.plot(numpy.ndarray), plt.plot(pandas Series), plt.plot(df)
- 但也可以用 Series.plot() 和 df.plot()
- 用 Numpy 畫圖：df['欄位'].values 變成 numpy.ndarray, 在用 plt.plot(ndarray) 畫圖
- 用 Pandas Series 畫圖：plt.plot(pd_series), pd_Series.plot(), df['欄位'].plot()
- 用 Pandas DataFrame 畫圖：df.plot(), plt.plot(df) 畫全部的欄位在同張圖
- df.plot(subplots=True) 把不同欄位畫在不同的 subplot 上，不需要自己切割子圖
- plt.yscale('log') 改成 log scale
- df['欄位'].plot(color='b', style='.-', legend=True) 可以對圖形做各種設定
- plt.axis((xmin, xmax, ymin, ymax))
plt.savefig('figure.png/jpg/pdf') 存圖檔
- exploratory data analysis (EDA)
- df.plot(kind='hist', x='欄位名' y='欄位名'), df.plot.hist(), df.hist() 都是畫 histogram 但是畫出來的結果有點差別，scatter, box 等圖也類似用法
- df.plot(kind='line/scatter/box/hist/area', x='欄位名', y='欄位名')
- 不指定 kind 預設就是畫 line plot
畫 scatter plot 時用 s=[點的大小] 來指定每個點的大小
- subplots=True 可以分成兩個圖，可以在畫 box plot 時使用
- 畫 hist 時 bins=數目 指定幾個 bin，range=(low, high) 指定範圍
- 畫 hist 的 PDF 時要加上 normed=True 畫 CDF (cumulative density functions) 要加上 normed=True, cumulative=True
- fig, axes = plt.subplots(nrows=2, ncols=1) 自己切割子圖
- 有用 subplots() 時，要用 ax=axes[n] 來指明畫在哪一張 subplot 上
df.fraction.plot(ax=axes[0], kind='hist', normed=True, bins=30, range=(0,.3))
- 畫 PDF，ax=axes[0] 指明畫在第一列
- df.fraction.plot(ax=axes[1], kind='hist', normed=True, cumulative=True, bins=30, range=(0,.3)) 畫 CDF，ax=axes[1] 指明畫在第二列
- df.mean() 算每個欄位的平均值
- df.mean(axis='columns') 一個列有很多欄位，axis='columns' 就是用該列全部欄位來算此列的平均值
- df.quantile(q) q 可以是介於0~1的數值或是列表
- 例如：df.quantile([0.05, 0.95]), 而 df.quantile(0.5) 等同於 median
- us = df[df['origin'] == 'US'] 會選出 origin 欄位的值是 US 的所有列，放到新的 DataFrame
- titanic.loc[titanic['pclass'] == 1].plot(ax=axes[0], y='fare', kind='box')

- 日期時間格式 yyyy-mm-dd hh:mm:ss
my_datetimes = pd.to_datetime(date_list, format='%Y-%m-%d %H:%M') date_list 是 str 的格式，用 .to_datetime 轉成 datetime64 的格式的 pandas.core.indexes.datetimes.DatetimeIndex
- time_series = pd.Series(temperature_list, index=my_datetimes)
ts4 = ts2.reindex(ts1.index, method='ffill') 把 ts2 的 index 用 ts1 的 index 取代，method=ffill (forward fill) 也可以用 method='bfill' (backward fill)

- 日期時間可以 resample，分成 down-sampling 和 up-sampling
series.resample(頻率).統計函數()
down-sampling 是把日變週，週變月，月變年等等，up-sampling 是把年變月，月變週，週變日等等
例如：df = df['Temperature'].resample('6h').mean() 算六小時的平均
例如：df = df['Temperature'].resample('D').count() 算每日的數目
- the concept of 'method chaining': df.method1().method2().method3()
- Rolling means (or moving averages) are generally used to smooth out short-term fluctuations in time series data and highlight long-term trends.
hourly_data.rolling(window=24).mean() would compute new values for each hourly point, based on a 24-hour window stretching out behind each point. The frequency of the output data is the same: it is still hourly.
- df.columns.str.strip() 欄位名字有空格就拿掉空格
- df['欄位名'].str.upper() 把該欄位所有儲存格都改大寫
- df['欄位名'].str.contains('某字串') 若是欄位中的儲存格找到某字串則該儲存格傳回 True
- ts2_interp = ts2.reindex(ts1.index).interpolate(how='linear')

- 設定時區：times_tz_central = times_tz_none.dt.tz_localize('US/Central')
- 轉換時區：times_tz_pacific = times_tz_central.dt.tz_convert('US/Pacific')

- 把格式改成 datetime64: df.Date = pd.to_datetime(df.Date)
- 把索引改成 datetime64: df.set_index('Date', inplace=True)

- 刪除某些欄位：df_dropped = df.drop(list_to_drop, axis='columns')

- 把數字的字串改成數值：pd.to_numeric(pd_series, errors='coerce') errors='coerce' 使得非數字的字串在轉成數值時變成 NaN 而不會出錯
- 例如：df_clean['dry_bulb_faren'] = pd.to_numeric(df_clean['dry_bulb_faren'], errors='coerce')
- daily_temp_climate = daily_climate.reset_index()['Temperature']
