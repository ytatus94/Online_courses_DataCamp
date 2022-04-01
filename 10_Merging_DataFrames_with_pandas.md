# 10. Merging DataFrames with pandas
* 讀取很多檔案：
  * 有很多檔案要讀的話，可以把檔案寫成一個 `list` 然後用 `for` loop 讀取
  * 或是用 list comprehension 讀取
  * 用通配符讀取
```python
# 用 for loop
dfs = []
for f in [f1, f2, f3]:
    dfs.append(pd.read_csv(f))
    
# 用 list comprehension
dfs = [pd.read_csv(f) for f in [f1, f2, f3]]

# 用通配符
from glob import glob
glob('sale*.csv')
```

* 
拷貝 df DataFrame： df_new = df_old.copy()
df.read_csv('檔案.csv', index_col='要當作 index 的欄位名字或 index') 如果 index 是時間格式的話可以加上 parse_dates=True 來使用 dateTime[64] 格式的 index
df.sort_index() 如果要改變升降冪排序的話就要有 ascending=False
df.sort_values(by='column name') 可以指定一照哪一行來排序
df.reindex([new index list]) 也可以用別的 DataFrame 的 index 當參數 df2.index，也可以再加上 .ffill() , .dropna() 等
可以直接用兩個 df 的名字來做+-*/數學運算，但是列與欄的數目要注意，有可能因為數目不同造成錯誤結果
temps_c.columns = temps_c.columns.str.replace('F', 'C') 把欄位名稱中的 F 改成 C
post2008 = gdp.loc['2008':] 選出所有 2008 與之後的日期的列
yearly = post2008.resample('A').last() 用年來 resample 新的結果用最後一個日期表示
yearly['growth'] = yearly.pct_change() * 100 百分比改變
pounds = dollars.multiply(exchange['GBP/USD'], axis='rows') 對每一列作運算，用 add(), multiply(), divid(), 比 +-*/ 好 axis='rows' 和 axis=0 是一樣的，也是預設值
df1.append(df2).append(df3) 可以把 Series 或 DataFrame 疊起來，但是 index 並不會更動，仍維持原先各自的 index，要再加上 .reset_index(drop=True) 才可以用新的 index drop=True 是直接修改原本的 index 欄位而不會產生一欄新的欄位保存舊的 index
pd.concat([s1, s2, s3], ignore_index=True) ignore_index=True 會忽略原本的 index 而採用 0, 1, 2,... 的 RangeIndex，預設是用 axis=0 (axis='rows') 來垂直(上下)連接 DataFrame 用 axis=1 (axis='columns') 來水平(左右)連接 DataFrame，可以加入參數 keys=[key1, key2, ...] 來指定多重 index 最外側的 index，keys 會是列或是欄則照 axis 指定，join='inner' 或 join='outer' (預設)
medals_sorted = medals.sort_index(level=0) 多重 index 做 sort 時要指定用哪個 index 來排序
medals_sorted.loc[('bronze','Germany')] 多重 index 做選取時，要用 tuple 把每一層的 index 刮起來
idx = pd.IndexSlice 多重 index 做切片時，要用 pd.IndexSlice
medals_sorted.loc[idx[:,'United Kingdom'], :] 多重 index 做切片的範例，這會選出所有第二層 index 是 United Kingdom 的列
slice_2_8 = february.loc['Feb. 2, 2015':'Feb. 8, 2015 ', idx[:, 'Company']]
pd.merge(df1, df2), pd.merge(df1, df2, on=['column1', 'column2', 'column3']), pd.merger(df1, df2, left_on=['df1_column1', 'df1_column2'], right_on=['df2_column1', 'df2_column2'], how='left/right/inner/outer', suffixes=['_left','_right'], fill_method='ffill') 沒有 on 的話就是依照兩個 DataFrame 中相同的欄位來合併
df1.join(df2, how='left/right/inner/outer')
pd.merge_ordered() 其實就是 pd.merge(df1, df2, how='outer').sort_values('Date')

pd.merge_asof(df1, df2, left_on='df1_column', right_on='df2_column') 有不太一樣的功能，不過我搞不懂

merged = pd.merge_asof(auto, oil, left_on='yr', right_on='Date')

yearly = merged.resample('A', on='Date')[['mpg', 'Price']].mean()

print(yearly.corr()) shows the Pearson correlation between the resampled 'Price' and 'mpg'.

fractions = medal_counts.divide(totals, axis='rows')

The expanding mean provides a way to see this down each column. It is the value of the mean with all the data available up to that point in time.

mean_fractions = fractions.expanding().mean()

hosts.loc[hosts.NOC.isnull()]

# Make bar plot of change: ax
ax = change.plot(kind='bar')

# Customize the plot to improve readability
ax.set_ylabel("% Change of Host Country Medal Count")
ax.set_title("Is there a Host Country Advantage?")
ax.set_xticklabels(editions['City'])
