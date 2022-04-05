# 4. Python Data Science Toolbox (Part 2)

## Iterable 和 iterator:
* An **iterable** is an object that can return an iterator, while an **iterator** is an object that keeps state and produces the next value when you call `next()` on it.
* `iterator = iter(iterable)` iterable 可以是 tuple, list, dict, range, file 等等的。
* `next(iterator)` 會產生下一個的值
  * 檔案也可以是 iterable，`next(檔案)` 就會一次讀一行
* `*iterator` 會一次就 iterate 全部

## Enumerate
* `enumerate()` returns an enumerate object that produces a sequence of **tuples**, and each of the tuples is an **index-value pair**.
* `enumerate(list, start=n)` 指明 index 從 n 開始，不指定 n 時預設是從 0 開始

## Zip
* `zip()`, which takes **any number of iterables** and returns a zip object that is an **iterator of tuples**.
* `zip(list_a, list_b, list_c, ...)` 可以有很多個 list 當參數
* `for z1, z2 in zip(list_A, list_B):` 在 list\_A 和 list\_B 中的 **index 相同的元素會形成一組 tuple**，將這組 tuple 的值分配給 z1, z2，所以 `z1=list_A[i]`, `z2=list_B[i]`。
* 例如:

```python
list_A = ['a', 'b', 'c']
list_B = ['A', 'B', 'C']
for z1, z2 in zip(list_A, list_B):
    print(z1, z2)
```
輸出會是

```python
a A
b B
c C
```

* There is no unzip function for doing the reverse of what `zip()` does.
* 星號 `*zip_obj`: **unpacks an iterable** such as a list or a tuple into positional arguments in a function call.
* 範例:
  * `z1 = zip(list_A, list_B)` 產生的 z1 是一個 zip 物件，是由一堆 tuple 所組成，由 list\_A 和 list\_B 的元素按照相同的 index 順序倆倆組成一對 tuples
  * `*z1` 是把 zip 物件解開成一堆 tuple
  * `result1, result2 = zip(*z1)` 這邊 `*z1` 先把 zip 物件，變成一堆 tuple，再把這些 tuple zip 起來，如此可還原 zip 前的列表

## Chunksize
* `for chunk in pd.read_csv('tweets.csv', chunksize=10):` 使用 `chunksize` 限制每次讀取的大小，讀完一個 chunk 再讀下一個
* 範例:
```python
text_reader = pd.read_csv('file.csv', chunksize=100)
```
讀 `file.csv` ，讀入後的 `text_reader` 型態是 `pandas.io.parsers.TextFileReader`，可以想像成是一個 list，每個元素就包含了 `chunksize=100 的大小`，可以用 for 來 loop 來取得檔案全部的內容

## List comprehension
* list comprehension 有兩種:

```python
[ output expression for iterator variable in iterable if predicate expression ]
和
[ output if-else for iterator variable in iterable]
```

例如:

```python
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']
new_fellowship = [member for member in fellowship if len(member) >= 7]
```
會得到 `['samwise', 'aragorn', 'legolas', 'boromir']`

```python
new_fellowship = [member if len(member) >= 7 else "" for member in fellowship]
```
會得到 `['', 'samwise', '', 'aragorn', 'legolas', 'boromir', '']`

* list comprehension 可以寫成**巢狀結構**
  * 例如: `matrix = [[col for col in range(0, 5)] for row in range(0, 5)]`

* 除了 list comprehension 以外也有 dict comprehensions 方法類似
* use of parentheses `()` in **generator** expressions and brackets `[]` in **list comprehensions**.
  * list comprehension 是**一次產生全部**並放在記憶體，數量大時內佔空間，而 **generator 是要用才產生**，數量大時不太佔記憶體空間。
  * generator function 和一般函數定義的方法一樣，差別在於 generator function 不是用 return 傳回而是用 `yield` 傳回
* `list(zip 物件)`, `dict(zip 物件)` 會將 zip 物件轉成 list 或是 dict，一但轉換了後 zip 物件的值**不再存在**，只剩下記憶體位置，不過裡面沒東西

## 開讀檔:

```python
with open('world_dev_ind.csv') as file:
    file.read() # 讀全部直到檔案結尾
    file.readline() # 一次只讀一行
    file.readlines() # 讀全部，每一行變成 list 的元素
```

* 用 `with` 開讀檔案的話，不用自己關掉檔案 `file.close()`
  * 用 `with XXX as YYY` 的方法叫做 context manager 
  * 只用 `file = open(檔案, mode)` 來開讀檔時，才要自己關掉檔案 `file.close()`

## DataFrame
* 選出符合條件判斷的部份當新的 DataFrame

```python
df_new = df_old[ df_old 條件判斷 ]
```  
  * 例如:
  
  ```python
  df_pop_ceb = df_urb_pop[ df_urb_pop['CountryCode'] == 'CEB' ]
  ```
  選出 `CountryCode` 是 `CEB` 的**列**，變成新的 DataFrame `df_pop_ceb`
* 用 DataFrame 呼叫 plot 來畫圖

```python
df.plot(kind='哪一種圖', x='欄位名', y='欄位名')
```

`kind` 可以是 `scatter`, `box`, `hist` 等，x 軸和 y 軸直接使用要用來畫圖的欄位的名字
  * 例如: 
  
  ```python
  data.plot(kind='scatter', x='Year', y='Total Urban Population')
  ```
  
  用 Year 當 x，Total 當 y 畫 scatter plot
