# 1. Intro to Python for Data Science

* 用 `list(列表的名字)` 或是 `list_name[:]` 來拷貝列表的話，是拷貝列表中的值，而非記憶體位置

## String

```python
str.upper() # 將 str 改成大寫
str.count("a") # 計算 a 在 str 中出現的次數
```

## 常用的 list 方法：

```python
list.index(a) # 找出列表中 a 的 index
list.count(a) # 計算列表中 a 出現的次數
list.append(a) # 將 a 加入列表
list.remove(a) # 把 a 從列表中移除
list.reverse() # 把列表元素反向排列
``` 

## NumPy
* `import numpy as np`

### NumPy array:
* `np.array([list])` 建立 array 型態是 `numpy.ndarray`，ndarray 表示 N-dimension array
* NumPy array 可以看成是 Python list 在 NumPy 裡面等價的東西，但用來建立 NumPy array 的 list 的元素必須要是相同型態。
如果用來建立 NumPy array 的 list 的元素型態不同，NumPy 會把所有元素轉成 string 型態
* 對 NumPy array 做加減乘除等操作就是對 NumPy array 中的每個元素操作，拿兩個 NumPy array 做運算，會用相對應的元素來計算
* NumPy array 切片的方式和 list ㄧ樣
* `np_array[ np_array 條件判斷 ]` 會選出符合條件判斷的元素
* `np_array.shape` 看陣列的維度，shape 是屬性不是方法所以沒有括號
* `np_array_2D[row][column]` 或 `np_array_2D[row, column]` 可存取二維 NumPy array 中的元素

### NumPy 統計:

```python
np.mean(np_array) # 平均值

np.mean( np_array 條件判斷 ) # 傳回滿足條件判斷的百分比

np.median(np_array) # 中位數

np.var(np_array) # variance

np.std(np_array) # 標準差

np.corrcoef(np_array1, np_array2) # 算 Pearson correlation coefficient

np.cov(np_array1, np_array2) # 算 covariance，會傳回一個 covariance_matrix

np.sqrt() # 算平方根

np.round(數值, 四捨五入到第幾位) # 四捨五入
```
  * 例如：
      
    ```python
    np.round(np.random.normal(mean, std, # of samples), 2)
    ```
    * 把第一個參數取到小數點第二位，
    * 第一個參數 `np.random.normal(mean, std, # of samples)` 是產生隨機的高斯分佈

* `np.column_stack( (list1, list2) )` 將兩個 list 寫成 column 向量後從左往右排在一起
  * Numpy 有 `vstack()`, `column_stack()`, `hstack()` 方法
  * 例如：
      
    ```python
    np.vstack(([1,2,3],[4,5,6]))
    = array([[1, 2, 3],
             [4, 5, 6]])
    ```
    ```python
    np.column_stack(([1,2,3],[4,5,6]))
    = array([[1, 4],
             [2, 5],
             [3, 6]])
    ```
    ```python
    np.hstack(([1,2,3],[4,5,6]))
    = array([1, 2, 3, 4, 5, 6]
    ```

---

* `help(func)` 和 `?func` 都可以用來查詢函數的資訊
  * 例如：`help(max)` 或是 `?max` 可以查 `max()` 函數的資訊
* `from math import pi`
* `from math import radians`
* `type(物件)` 查物件的型態
