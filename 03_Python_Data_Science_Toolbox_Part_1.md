# 3. Python Data Science Toolbox (Part 1)

* `import builtins` builtin 模組不是 builtin 的，要 import 才行。
* `dir(builtins)` 顯示所有 builtin 函數。

## Scope
* `global var_name` 將 var_name 變成全域變數。
* `nonlocal var_name` 將巢狀函數中的 var_name 變成函數內共享的。

## 函數
* 函數可以是巢狀的，巢狀外側的函數可以傳回一個巢狀內側的函數。
* 函數的參數一個 `*` 是 tuple 兩個 `*` 是 dict
  * `def func(*args)`: variable-length arguments (*args)
  * `def report_status(**kwargs)`: variable-length keyword arguments (**kwargs)
    * 這邊都說是 keyword 參數了，就是用 dict
* map
```python
map(lambda func, list) # 將 list 中的每個元素做 lambda function 的計算，傳回一個 map 物件。
list(map(lambda func, list)) # 
```

* filter
```python
filter(lambda func, list) # 將 list 中的每個元素做 lambda function 的計算，傳回一個 filter 物件。
list(filter(lambda func, list)) # 把 filter 物件轉成 list
```

* reduce
```python
from functools import reduce # reduce 函數是在 functools 模組裡面
reduce(lambda func, list)
```

## 例外處理：
```python
if condition:
    raise ValueError('...')
try:
    要幹什麼事情
except: <---這邊也可以用 except 某例外Error:
    發生例外了怎麼辦
```
