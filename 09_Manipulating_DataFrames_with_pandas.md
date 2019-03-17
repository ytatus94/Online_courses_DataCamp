9. Manipulating DataFrames with pandas

- df['欄位名']['列名'] 和 df.欄位名['列名'] 得到的結果是一樣的，df.loc['列名', '欄位名'] 和 df.iloc[列索引, 欄索引] 傳回可能是一個數值，一個 Series 或是一個 DataFrame，df[['欄位名1', '欄位名2']] 傳回的才是一個 DataFrame

- 各種選擇的方法：
df.loc['A', 'B'] 選擇 A 列 B 欄的元素
- df.loc['A':'B',:] 選擇 A 到 B 列，包括 B 列，全部的欄位的元素
- df.loc[:, 'A':'B'] 選擇 A 到 B 欄，包括 B 欄，每一列的元素
- df.loc['A':'B':-1] 有第三個元素 -1 表示將結果逆排序
- df.loc[['A', 'B'], ['C', 'D']] 選擇 A 列和 B 列 和 C 欄和 D 欄，傳回的是一個 DataFrame
- 用 list 和 列與欄位名的方式可以混和使用
df.iloc[] 的方法相同，只是改成用數值索引

- 用 df.loc[‘A':'B', ['C', 'D']] 的話是包含 B 欄位的，但是 df.iloc[a:b, [c,d]] 就不包含 b
- df[df 的條件判斷], df['欄位名'][df 的條件判斷] 多格條件判斷時每一個都用 () 括起來，用 & 和 |
- 可以對條件判斷後選出的資料做運算

- 可以用 df.欄位1[df.欄位2 的條件判斷] 依照欄位 2 判斷的結果選擇欄位 1 ，還可以將結果做運算，會直接反映在 df 上
- 例如：df.eggs[df.salt > 55] += 5 和 election['winner'][too_close] = np.nan
- df2 = df.copy() 拷貝 df DataFrame 到 df2
- df.all() 選擇所有不含有 0 的欄位，只要該欄位有 0 就不要 (NaN 不是 0)
- df.any() 選擇所有非 0 值的欄位，只要該欄位有非 0 就要
- df.isnull().any() 選擇所有含有 NaN 的欄位
- df.notnull().all() 選擇所有不含有 NaN 的欄位
- df.dropna(how='any') 刪除任何含有 NaN 的列
- df.dropna(how='all') 刪除整列都是 NaN 的列
- df.dropna(thresh=1000, axis='columns') 刪除任何欄位中非零值的個數小於1000的
- df.floordiv(n) 和 np.floor_divide(df, n) 一樣，對 df 內的所有元素做整數除法，除數是 n
- df.apply(自訂函數或是 lambda 函數) 會對全部欄位的資料做 apply() 內運算
- apply() 內放自訂函數名就好，不用加上自訂函數的參數
- df.index 傳回 index 的值
- df.index.str.upper()
df.index.map(str.lower) index 欄位不可以用 apply()，要用 map()
- df.columns = ['欄位1', '欄位2'] 可以對欄位命名或重新命名
- .map() method is used to transform values according to a Python dictionary look-up.
- 例如：
red_vs_blue = {'Obama':'blue', 'Romney':'red'}
election['color'] = election['winner'].map(red_vs_blue)
- .apply() 和 .map() 的效率比較差，用 vectorized function 的效率會比較好
NumPy, SciPy, pandas 都有 vectorized function
NumPy 中的 vectorized function 叫做 Universal Functions 或 UFuncs
- In statistics, the z-score is the number of standard deviations by which an observation is above the mean - so if it is negative, it means the observation is below the mean.
z-score 是計算 “和 mean 的 deviation 是幾個標準差”
the zscore UFunc will take a pandas Series as input and return a NumPy array.
例如：
from scipy.stats import zscore
turnout_zscore = zscore(election['turnout']) 計算 turnout 欄位的 z-score
- election['turnout_zscore'] = turnout_zscore 再把計算結果賦值到新的欄位
- Index: sequence of labels，是 immutable，資料型態一致
Series: 1D array with index
DataFrame: 2D array with Series as columns
- pd.Series([值的列表], index=[index 的列表]) 可以建立 Series
- 若在建立時沒有指定 index=[index 的列表] 則預設會使用從 0 開始的數值索引
- 如果事後想改 index 可以用 s.index = [index 的列表] 但是只能一次改全部 (series 是 immutable 的，所以沒有可以只改某一個 index 的方法)
- series.index.name = '列名' 可以為 index 欄位命名
- df.columns.name = '欄名' 可以為欄位那一列命名
- 欄名        col1       col2
列名
index1    value1   value2
index2   value3  value4
df.index.name 和 df.columns.name 若不加等號賦值就只是顯示結果
- df.index 顯示 index
- df.index = [index 的列表] 可以將 index 改成列表中指定的
- df.index = df['欄位1'] 可以將 欄位1 指定為 index
- 上面的方式會造成欄位1重複，一個是原先的欄位另一個是用來當 index，可以使用 del df['欄位1'] 可以刪除原先的欄位1剩下 index
- df = pd.read_csv('file.csv', index_col='欄位1') 在讀入 csv 檔案時就指明用 欄位1 當 index，如果用 index_col=['欄位1', '欄位2', '欄位3'] 就換變成 multi-level row index
- Hierarchical index 就是有兩個或以上的 index，也叫做 Multi-level Index
用 df.info() 看型態的話 Hierarchical index 顯示的型態是 MultiIndex
df = df.set_index(['index1名', 'index2名']) 設定 Multi-level index
- df.index.names 顯示 index 的名字，因為有多個 index 所以這時候要用複數的 names
- df = df.sort_index() 為 Multi-level index 做排序

- 用 Hierarchical index 做選擇的方式有：
df.loc[('外層 index 名', '內層 index 名')] 或 df.loc[('外層 index 名', '內層 index 名'), 欄位名] 要用括號把 Multi-level index 括起來
- df.loc['外層 index 名'] 或 df.loc['外層 index 名1': '外層 index 名3'] 只用最外層的 index，列出符合選擇條件的結果
- df.loc[(['外層 index 名1', '外層 index 名3'], '內層 index 名'), 欄位名] 和 df.loc[('外層 index 名', ['內層 index 名1', '內層 index 名3']), :] 可以用列表選擇多個外層或內層 index
- df.loc[(slice(None), slice('內層 index 名1', '內層 index 名3')), :] 外層 index 要全選時，不能用 : 要用 slice(None)
- df.pivot(index='當成 index 的欄位', columns='繼續當欄位的', values='當值的欄位') 沒提供 values 的話，所有沒有列入 index 和 columns 的欄位都會被當成 values
- df.set_index(['outmost index', 'innermost index']) 把一個普通的單一 index 的 DataFrame 轉成 Multi-level index 的 DataFrame
- df.unstack(level='innermost index') 會把 Multi-level index 的 DataFrame 變成樞紐分析表 (pivot 表格)，也可以用 df.unstack(level=1) 達到同樣的結果，level 從是最外側的 index 當成 0 開始算起
- 用 df.stack(level='欄位') 可以將樞紐分析表 stack 成 Multi-level index 的 DataFrame，預設會把欄位變成 innermost index，可以藉由 df.swaplevel(0, 1) 把 outmost 和 innermost 欄位對調，但是對調之後的列是沒有排序的，可以用 df.sort_index() 將列作排序
- melt() 是用來把 pivot 表格轉換成最原本的單一 index 的 DataFrame 的格式，或者把 wide shape 變成  long shape 的格式
pd.melt(df, id_vars=['欄位'], value_vars=['欄位1', '欄位2'], var_name='欄位1', value_name='欄位2') id_vars 指定哪些欄位在 melt 之後仍然是原本的欄位，value_vars 指定哪些欄位變成了值，若沒有用 var_name 和 value_name 的話，欄位的名字會是 variable 和 value
- pd.melt(df, col_level=0) 會把 Multi-level index 的 df 中的 index 拿掉，只剩下 column 來做 melt()

- 當欄位內的值有重複的時後，就不能用 pivot() 要用 pivot_table()
df.pivot_table(index='欄位', columns='欄位', values='欄位', aggfunc='函數', margins=True)
- 要是沒有指定 aggfunc 預設會使用 mean 求平均值
如果有 margins=True 那就會在最下面多一列，計算各個欄位的總和
- df1.equals(df2) 會比較兩個 DataFrame 並傳回比較結果的布林值
- DataFrame 經過 groupby() 後的型態是 pandas.core.groupby.DataFrameGroupBy
- type(df.groupby().groups) 是個 dict，key 是 groupby() 的欄位，value 是該 group 對應的 row
- df.groupby('欄位').func()
- func() 可以是 mean(), std(), sum(), first(), last(), max(), min(), count() 等等的
- df.groupby('欄位1')['欄位2'].func()
- df.groupby('欄位1')[['欄位2', '欄位3']].func()
- df.groupby(['欄位', '欄位']).func() 產生的表格是 multi-level index 的
- df.groupby(s)['欄位'].func() 其中 s = pd.Series([列表])
- df.groupby('欄位1')[['欄位2', '欄位3']].agg(['func1', 'func2']) 使用兩個函數來做 aggregation，只用函數的名字，並且括起來，可以是用列表或是 tuple 的方式傳入 agg() 中，傳回的結果一個 multi-level column index 的 DafaFrame
- df.groupby('欄位1')[['欄位2', '欄位3']].agg(自訂func) 使用自訂的函數來做 aggregation，函數名字沒有括號，因為是 aggregation 所以自訂函數只能傳回一個值
- df.groupby('欄位1')[['欄位2', '欄位3']].agg({'欄位2': 'func1', '欄位3':自訂func}) 分別對兩個欄位使用不同的 aggregation 函數
- df['欄位'].unique()
- df['欄位'].astype('category') 用 category 會佔用較少記憶體，速度較快
- multi-level column index 的 DataFrame 可以用 df.loc[:, ('欄位1', '子欄位2')] 的方式來顯示
- multi-level row index 的 DataFrame 也可以用
- df.groupby(df.index.strftime('%a')) 把 datetime index 轉成星期幾，再 groupby() 起來
- Z score: 計算每一個數值和 mean 比較偏移了幾個標準差，通常超過 +/-3 的話就當成 outliers
def zscore(series):
    return (series - series.mean()) / series.std()
也可以直接呼叫 SciPy 裡面寫好的 from scipy.stats import zscore
- transform() applies a function element wise to group
- df.groupby('欄位1')['欄位2'].transform(zscore)
- 太複雜的函數不能用 transform() 要改用 apply()
df.groupby('欄位1').apply(更複雜的函數)
- groupby() 也可以拿來做 missing value imputation
例如：titanic.groupby(['sex', 'pclass'])['age'].transform(impute_median)
其中 def impute_median(series):
           return series.fillna(series.median())
- def disparity(gr):
    # Compute the spread of gr['gdp']: s
    s = gr['gdp'].max() - gr['gdp'].min()
    # Compute the z-score of gr['gdp'] as (gr['gdp']-gr['gdp'].mean())/gr['gdp'].std(): z
    z = (gr['gdp'] - gr['gdp'].mean())/gr['gdp'].std()
    # Return a DataFrame with the inputs {'z(gdp)':z, 'regional spread(gdp)':s}
    return pd.DataFrame({'z(gdp)':z , 'regional spread(gdp)':s})

- 若 groupby() 之後只要看某個特定的結果，就必須要先做 filtering 再做 aggregation
例如：
splitting = auto.groupby('yr')
for group_name, group in splitting:
    avg = group.loc[group['name'].str.contains('chevrolet'), 'mpg'].mean()
- 先依照 yr 做 groupby() 在迴圈中做 filter 選出 name=chevrolet 的對 mpg 求 mean()
也可以改成用 dictionary comprehension 寫：
chevy_means = {year:group.loc[group['name'].str.contains('chevrolet'),'mpg'].mean() for year,group in splitting}

- 另外一種 groupby() 後 filtering 的方式，不過結果會與上面不太一樣
chevy = auto['name'].str.contains('chevrolet')
auto.groupby(['yr', chevy])['mpg'].mean() 這個會把 name=chevrolet的當成 true 其他的當成 false，然後對 true 和 false 都做 mean()

- 也可以在 groupby() 之後用 .filter()
例如：sales.groupby('Company').filter(lambda g: g['Units'].sum() > 35) 這邊的 g 其實 pandas 就會用 sales.groupby('Company') 取代，但是顯示的結果只是 sales 這個 DataFrame 並且只有所有 company units sum > 35 的公司留下來
- filter() 先篩選出了符合條件的公司，然後 pandas 會把 sales 中的公司是符合要求的每一列顯示出來
- under10 = (titanic['age'] < 10).map({True:'under 10', False:'over 10'})
先用 (titanic['age'] < 10) 產生一個布林值的 pandas series 再用 map({key:value}) 把布林值改成其他的東西
- df.sum(axis='columns') 把同一個列中的資料，沿著欄位的方向相加
- df.sort_values('欄位', ascending=False) 把欄位做遞減排序
- df.drop_duplicates() 刪除重複的列資料
- idxmax(): Row or column label where maximum value is located
- idxmin(): Row or column label where minimum value is located
- 預設是傳回最大或最小的那一列的 index 如果要傳回的是欄位的 index 的話，要加上參數 axis='columns' 或是 axis=1
- df.T 可以把 DataFrame 轉置
- pandaSeries.nunique() 傳回有多少個唯一的值
- df['欄位'].isin(['值1', '值2']) 會選出欄位的值是值1或值2的列
- df.plot(kind='line', marker='.')
matplotlib 遇到 multi-level index DataFrame 的時候並沒辦法處理，所以要先把 DataFrame unstack()
- df.plot.area() 畫 area plot，就跟線圖差不多但是把某條線的結果疊在前一條線上並且用顏色填滿
- medals.Medal = pd.Categorical(values=medals.Medal, categories=['Bronze', 'Silver', 'Gold'], ordered=True) 把 medals['Medal'] 變成是 category，而且照著 categories 定義的順序排列
