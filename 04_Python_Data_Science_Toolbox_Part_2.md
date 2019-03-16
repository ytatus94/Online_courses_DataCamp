4. Python Data Science Toolbox (Part 2)

- an iterable is an object that can return an iterator, while an iterator is an object that keeps state and produces thefinext value when you call next() on it.
iterator = iter(iterable) iterable 可以是 tuple, list, dict, range, file 等等的。
- next(iterator)
*iterator 會一次就 iterate 全部
- 檔案也可以是 iterable，next(檔案) 就會一次讀一行
- enumerate() returns an enumerate object that produces a sequence of tuples, and each of the tuples is an index-value pair.
enumerate(list, start=n) 指明 index 從 n 開始，預設是從 0 開始
- zip(), which takes any number of iterables and returns a zip object that is an iterator of tuples.
zip(list_a, list_b, list_c, ...) 可以有很多個 list 當參數
- for z1, z2 in zip(list_A, list_B): 在 list_A 和 list_B 中的 index 相同的元素會形成一組 tuple，將這組 tuple 的值分配給 z1, z2，所以 z1=list_A[i], z2=list_B[i]。
- There is no unzip function for doing the reverse of what zip() does.
星號 * unpacks an iterable such as a list or a tuple into positional arguments in a function call.
- 範例：z1 = zip(mutants, powers) 產生的 z1 是一個 zip 物件，是由一堆 tuple 所組成，由 mutants 和 powers 的元素按照相同的 index 順序倆倆組成一對 tuples
- *z1 是把 zip 物件解開成一堆 tuple
- result1, result2 = zip(*z1) 這邊 *z1 先把 zip 物件，變成一堆 tuple，再把這些 tuple zip 起來，如此可還原 zip 前的列表
- for chunk in pd.read_csv('tweets.csv', chunksize=10): 使用 chunksize 限制每次讀取的大小，讀完一個 chunk 再讀下一個
- 用 text_reader = pd.read_csv('file.csv', chunksize=100) 讀 file.csv ，讀入後的 text_reader 型態是 pandas.io.parsers.TextFileReader，可以想像成是一個 list，每個元素就包含了 chunksize=100 的大小，可以用 for 來 loop 來取得檔案全部的內容
- list comprehension: [ output expression for iterator variable in iterable if predicate expression ] 和 [ output if-else for iterator variable in iterable]
- 例如：fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']
new_fellowship = [member for member in fellowship if len(member) >= 7]
會得到 ['samwise', 'aragorn', 'legolas', 'boromir']
new_fellowship = [member if len(member) >= 7 else "" for member in fellowship]
會得到 ['', 'samwise', '', 'aragorn', 'legolas', 'boromir', '']
- list comprehension 可以寫成巢狀結構
- 例如：matrix = [[col for col in range(0, 5)] for row in range(0, 5)]

- 除了 list comprehension 以外也有 dict comprehensions 方法類似
- use of parentheses () in generator expressions and brackets [] in list comprehensions.
- list comprehension 是一次產生全部並放在記憶體，數量大時內佔空間，而 generator 是要用才產生，數量大時不太佔記憶體空間。generator function 和一般函數定義的方法一樣，差別在於 generator function 不是用 return 傳回而是用 yield 傳回
- list(zip 物件), dict(zip 物件) 會將 zip 物件轉成 list 或是 dict，一但轉換了後 zip 物件的值不再存在，只剩下記憶體位置，不過裡面沒東西

- 開讀檔：
with open('world_dev_ind.csv') as file:
    file.readline() 一次只讀一行
-     file.read() 讀全部直到檔案結尾
- 用 with 開讀檔案的話，不用自己關掉 file.close()
- 只用 file = open(檔案, mode) 來開讀檔時，才要自己關掉 file.close()
- df_new = df_old[ df_old 條件判斷 ]  選出符合條件判斷的部份當新的 DataFrame
- 例如：df_pop_ceb = df_urb_pop[ df_urb_pop['CountryCode'] == 'CEB' ]
- df.plot(kind='哪一種圖', x='欄位名', y='欄位名') 直接用 DataFrame 呼叫 plot 來畫圖，kind 可以是 scatter, box, hist 等，x 軸和 y 軸直接使用要用來畫圖的欄位的名字
- 例如：data.plot(kind='scatter', x='Year', y='Total Urban Population')
