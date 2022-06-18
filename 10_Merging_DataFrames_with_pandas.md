# 10. Merging DataFrames with pandas

## 用 Pandas 讀檔
* 讀取一個檔案: `df.read_csv('檔案.csv', index_col='要當作 index 的欄位名字或 index') `
  * 如果 index 是時間格式的話可以加上 `parse_dates=True` 來使用 `dateTime[64]` 格式的 index

* 讀取很多檔案：
  * 有很多檔案要讀的話，可以把檔案寫成一個 `list` 然後用 `for` loop 讀取
  * 或是用 list comprehension 讀取: `dfs = [pd.read_csv(f) for f in [f1, f2, f3]]`
  * 用通配符讀取

```python
# 用 for loop 讀取
dfs = []
for f in [f1, f2, f3]:
    dfs.append(pd.read_csv(f))
    
# 用通配符讀取
from glob import glob
glob('sale*.csv')
```

* 拷貝 DataFrame: `df_new = df_old.copy()`
* 排序:
  * 依照 index 來排序: `df.sort_index()`, 預設是降冪排序 (`ascending=True`)，如果要改變成升降冪排序的話就要有 `ascending=False`
  * 依照某欄位來排序: `df.sort_values(by='column name') `
* 改變 DataFrame 的 index: `df.reindex([new index list])`
  * 也可以用別的 DataFrame 的 index 當參數，例如 `df.reindex(df2.index)`
  * 改變完之後也可以再加上 `.ffill()` , `.dropna()` 等
* 可以直接用兩個 df 的名字來做 + - * / 等數學運算，但是列與欄的數目要注意，有可能因為數目不同造成錯誤結果
  * 用 `add()`, `multiply()`, `divid()`, 比用 + - * / 好
* 參數 `axis='rows'` 和 `axis=0` 是一樣的，也是預設值
  * `axis=columns` 和 `axis=1` 一樣 
* `temps_c.columns = temps_c.columns.str.replace('F', 'C')` 把欄位名稱中的 F 改成 C
* `post2008 = gdp.loc['2008':]` 選出所有 2008 與之後的日期的列
* `yearly = post2008.resample('A').last()` 用年來 resample 新的結果用最後一個日期表示
* `yearly['growth'] = yearly.pct_change() * 100` 百分比改變
* `pounds = dollars.multiply(exchange['GBP/USD'], axis='rows')` 對每一列作運算，用 `add()`, `multiply()`, `divid()`, 比 + - * / 好 


## 多重 index
* 多重 index 可能是 row index 是 hierarchical 的，也可能是 column index 是 hierarchical 的
* 排序: `df_sorted = df.sort_index(level=要用來排序的那個 index)`
  * 對有著多重 index 的 DataFrame 做 sort 的時候，要指定用哪一個 index 來排序，最外側的 index 是 `0`
* 選取: `df_sorted.loc[("index_1", "index_2")]`
  *   對有著多重 index 的 DataFrame 做選取時，要用 tuple 把每一層的 index 刮起來
* 切片: 
  * 多重 index 做切片時，要先用 `idx = pd.IndexSlice` 選出 index
  * 範例 1: `df_sorted.loc[idx[:,'index_2'], :]` 這會選出所有第二層 index 是 `index_2` 的列
  * 範例 2: `df2 = df.loc["index_1" : "index_2", idx[:, "某欄位"]]` 
* `slice_2_8 = february.loc['Feb. 2, 2015':'Feb. 8, 2015 ', idx[:, 'Company']]`

## 合併 DataFrame
* `df1.append(df2).append(df3)` 可以把 Series 或 DataFrame 疊起來，但是 index 並不會更動，仍維持原先各自的 index
  * 要再加上 `.reset_index(drop=True)` 才可以用新的 index
    * `drop=True` 是直接修改原本的 index 欄位而不會產生一欄新的欄位保存舊的 index
* `pd.concat([s1, s2, s3], ignore_index=True)`
  * `ignore_index=True` 會忽略原本的 index 而採用 0, 1, 2,... 的 RangeIndex
    * 預設是用 `axis=0` (`axis='rows'`) 來垂直(上下)連接 DataFrame 
    * 用 `axis=1` (`axis='columns'`) 來水平(左右)連接 DataFrame
    * 可以加入參數 `keys=[key1, key2, ...]` 來指定多重 index 最外側的 index，keys 會是列或是欄則照 axis 指定
    * `join='inner'` 或 `join='outer'` (預設)

* 各種 merge:

```python
pd.merge(df1, df2) # 沒有 `on` 的話就是依照兩個 DataFrame 中相同的欄位來合併
pd.merge(df1, df2, on=['column1', 'column2', 'column3'])
pd.merger(df1, df2, left_on=['df1_column1', 'df1_column2'], right_on=['df2_column1', 'df2_column2'], how='left/right/inner/outer', suffixes=['_left','_right'], fill_method='ffill')

df1.join(df2, how='left/right/inner/outer')
# join 也是一種 merge，是依照 df1 和 df2 的 index 做 merge

# 對 time series data 合併並且排序
pd.merge_ordered() # 其實就是 pd.merge(df1, df2, how='outer').sort_values('Date')

# 是 left-join 的變形，用最接近的 key 來合併而不是用相同的 key 來合併
pd.merge_asof(df1, df2, left_on='df1_column', right_on='df2_column') # df1 和 df2 在合併前必須已經用 key 排好序了
```

* 範例:

```python
merged = pd.merge_asof(auto, oil, left_on='yr', right_on='Date')
yearly = merged.resample('A', on='Date')[['mpg', 'Price']].mean()
print(yearly.corr()) # shows the Pearson correlation between the resampled 'Price' and 'mpg'.
```



* `fractions = medal_counts.divide(totals, axis='rows')`

* The expanding mean provides a way to see this down each column. It is the value of the mean with all the data available up to that point in time.

* `mean_fractions = fractions.expanding().mean()`

* `hosts.loc[hosts.NOC.isnull()]`

* 畫圖

```python
# Make bar plot of df
ax = df.plot(kind='bar')

# Customize the plot to improve readability
ax.set_ylabel("Y 軸的名字")
ax.set_title("圖片的 Title")
ax.set_xticklabels(editions['City'])
```