# 5. Importing Data in Python (Part 1)

* 在 IPython 中用 `!` 開頭的話可以使用 shell command

## 開讀檔:
* 開檔 `file = open('filename.txt', mode='r')` 
  * 只讀取檔案時用 `mode='r'` 寫入檔案時使用 `mode='w'`
  * 檔案讀完要記得 `file.close()` 關掉，
* `with open('filename.txt', 'r') as file:` 
  * 用 `with` 來開讀檔的話，不必用 `file.close()` 關掉
  
```python
file.read() # 一次讀全部
file.readline() # 一行一行讀
file.readlines() # 一次讀全部，傳回一個 list，每一行是 list 的一個元素

file.close() # 關閉檔案。
file.closed # 確認檔案是否關閉， closed 是屬性，不是方法，所以沒有括號。
```

## Zen of Python
```python
import this
```

## 用 Numpy 來讀檔:
* 用 `loadtxt()` 或 `genfromtxt()` 或 `recfromcsv()`
* 例如:

```python
data = np.loadtxt(file, delimiter=',', skiprows=1, usecols=[0, 2], dtype=str)
```
把 file 讀入 Numpy array

`delimiter=','` 指定用逗號當分隔符號

不讀 `skiprows` 指明的列數 (不是列 index)

只讀入 `usecols` 選定的欄位

`dtype=str` 指明讀入的是 `str`，因為**預設是讀成數字**，如果有字串的話，讀取就會錯誤，因此才要指明讀入的是 `str`

讀入後 data 的型態是 `numpy.ndarray`

* 例如:

```python
data = np.loadtxt("file.txt", delimiter='\t', dtype=float, skiprows=1)
```

用 `\t` 當分隔符號，指明讀入的是 `float` 型態，不讀入檔案中的第一列

* 例如:

```python
data = np.genfromtxt('file.csv', delimiter=',', names=True, dtype=None)
```
用 `names=True` 表示有 header，`dtype=None` 自動決定每一欄的型態

* 例如: `np.recfromcsv(file)` 相當於 `np.genfromtxt()` 用 `delimiter=',', names=True, dtype=None` 參數
* `np.shape(data)` 可以看 data 的維度

---

```python
im_sq = np.reshape(im, (28, 28)) # 改變 im 的維度成為 28 x 28，其中 im 是一個 numpy.ndarray
plt.imshow(im_sq, cmap='Greys', interpolation='nearest')
plt.scatter(data_float[:, 0], data_float[:, 1])
```

---

## 用 Pandas 來讀檔

* 用 `read_csv()` 或 `read_table()`

```python
df = pd.read_csv('file.csv') # 讀成 DataFrame
df = pd.read_csv('file.csv', nrows=5, header=None) # 只讀前五列，沒有 header
df = pd.read_csv(file, sep='\t', comment='#', na_values=['Nothing'])
```
用 `sep` 指明 delimiter 是 `\t`

`comment` 指明註解行的開頭是什麼字元，不會讀入註解行

`na_values` 用一個列表來指定什麼東西要讀入成 `NA`/`NaN`

* 用 `\t` 當分隔符的有時候是寫成 tsv 檔案

```python
df.head() # 顯示前五列
df.tail() # 顯示最後五列
```
* `df_array = df.values` 把讀入的 DataFrame 轉成是 Numpy array
* `pd.DataFrame.hist(data[['Age']])` 等同於 `data[['Age']].hist()` 注意要用雙括號變成 Pandas DataFrame

---

```python
import os
wd = os.getcwd() # get current working directory
os.listdir(wd)
```

## 用 Pickle 來讀檔
* 用來存取 python 的  bytestream 格式的檔案

```python
import pickle 
with open('data.pkl', 'rb') as file: # 第二個參數 rb 表示 read only, binary
    d = pickle.load(file) # 會把 d 讀成一個 dict
```

## 用 Pandas 讀 Excel 檔
xl = pd.ExcelFile(file) 讀入 Excel 檔，xl 的格式是 pandas.io.excel.ExcelFile
- xl.sheet_names 顯示 Excel 檔中全部的 worksheet 的名字，是一個 list
- df = xl.parse('worksheet 的名字') 把 worksheet 讀入 DataFrame，可以用 worksheet 的名字或是 index 指定讀入哪個 worksheet
- 例如：df = xl.parse(0, skiprows=[1], names=['Country', 'AAM due to War (2002)']) 用 index 指明要讀第零個 worksheet，要 skip 的 row 還有為讀入的欄位命名
- 例如：df = xl.parse(1, parse_cols=[0], skiprows=[1], names=['Country']) 用 parse_cols 指明要讀哪個欄位，parse_cols, skiprows, names 都是用 list 來指定

## 用 Pandas 來讀 SAS/Stata 檔：
from sas7bdat import SAS7BDAT 用來讀取 SAS/Stata
- with SAS7BDAT('input.sas7bdat') as file:
-     df_sas = file.to_data_frame() 讀 sas 檔案 格式是 pandas.core.frame.DataFrame
- df = pd.read_stata('disarea.dta') 讀 Stata 檔案 格式是 pandas.core.frame.DataFrame

## 用 Pandas 來讀 H5PY 檔：

```python
import h5py
h5py_data = h5py.File(h5py_file, 'r') # 讀 HDF5 檔案
```
h5py_data 的格式是 `h5py._hl.files.File`，其實是 dict of dict 的格式，裡面有一堆 keys，可以用 `data['key1']['key2'].value` 取值

* `np.arange(start, stop, step)` 產生一個 numpy.ndarray 但是不包含 stop 的值

## 用 Pandas 來讀 Matlab 檔

```python
import scipy.io
mat = scipy.io.loadmat('Matlab.mat') # 讀入 Matlab 檔案，讀入後的格式是 dict
```
讀入的 Matlab 檔案是 dict，某個 key 對應的 value 是 Numpy ndarray

```python
spicy.io.savemat('file.mat') # 存成 Matlab 檔案
fig = plt.figure()
```

## 讀 SQL 檔：
from sqlalchemy import create_engine
engine = create_engine('sqlite:///檔案.sqlite') 要先和 SQL database 建立連結的介面
- table_names = engine.table_names() 取得資料庫中全部的表格的名字
- con = engine.connect() 這邊才是真的建立連結
- rs = con.execute('SELECT * FROM Album') 執行 SQL 指令，並把結果存到 rs
- df = pd.DataFrame(rs.fetchall()) 把 rs 的全部結果存到 DataFrame
- con.close() 關閉連結
- 也可以使用 context manager 的方式：
- with engine.connect() as con:
    rs = con.execute('SELECT LastName, Title FROM Employee')
-     df = pd.DataFrame(rs.fetchmany(size=3)) 用 fetchmany(size=3) 只讀取 rs 中的三筆資料
-     df.columns = rs.keys() 用 rs.key() 得到表格欄位的名字，並用來命名 DataFrame 欄位
- 用 context manager 的話就不必自己關閉連結
直接用 Pandas 讀取 SQL：df = pd.read_sql_query('SELECT * FROM Album', engine)
- df.equals(df1) 比較兩個 DataFrame 是否相同，回傳布林值
