# Data Analysis in Excel

* 資料分析的過程: Business Question --> Get Data --> Explore Data --> Prepare Data --> Analyze Data --> Present Findings
* 基本操作
  * `CMD/CTRL + Arrow key`: 跳到該方向的下一個空白儲存格
  * `Page Up/Down/Left/Right`: 跳一頁
  * Go To: `CMD/CTRL + G`
  * Find: `CMD/CTRL + F`
  * 選擇 range 時用 A:A 表示整個 A column
* 有四種資料型態
  * DATE: 靠右對齊
  * NUMBER: 靠右對齊
  * BOOLEAN: 置中對其
  * TEXT: 靠左對齊
* 實用內建函數
  * `EXACT(value1, value2)`: 比對 value1 和 value2，相同的話是 TRUE 不同就是 FALSE
  * `TRIM(value)`: 移除多餘的空格
  * `SORT(range, [sort_index], [sort_order])`: 依照 sort_index 來排序 range
    * sort_order=1 是遞增排序
    * sort_order=-1 是遞減排序
  * `FILTER(range, criteria, [if_empty])`: 依照 creteria 來篩選 range 範圍內的資料
    * if_empty 指定如果是空值的時候要幹嘛
  * `TEXT(value, format)`: 型態轉換，把數值轉換成字串型態，就放一堆 0
    * 0的數目表示幾位數，例如: "000000" 就是指定六位數
  * `VALUE(text)`: 型態轉換，把字串型態轉換成數值
  * `ROUND(VALUE(text), decimal place)`: 四捨五入
  * `CONCATENATE(value1, [value2], ...)`: 連結 value1 和 value2 等等的
  * `LOWER(text)`: 全部轉小寫
  * `UPPER(text)`: 全部轉大寫
  * `PROPER(text)`: 轉成首字大寫
  * `LEN(text)`: 計算字串長度
  * `LEFT(text, number)`: 從字串最左邊開始取 number 個字元
  * `RIGHT(text, number)`: 從字串最右邊開始取 number 個字元
  * `SUBSTITUTE(text, old_text, new_text, [instance])`: 在字串 text 中把 old_text 用 new_text 取代
  * `date1 - date2`: 可以計算兩個日期差了幾天
  * `NOW()`: 現在時刻
    * 包含時間幾點幾分等等
  * `TODAY()`: 今天日期
    * 只有日期沒有時間
  * `MONTH(date)`: 取出 date 的月份，用 1 到 12 表示
  * `WEEKDAY(date, [return_type])`: 顯示 date 的日子是一週的哪一天
    * return_type=1 表示 sunday 是第一天 (預設值)
    * return_type=2 表示用 Monday 當第一天
  * `VLOOKUP()` 是最重要的內建函數之一了
    * V 表示 vertical
    * `VLOOKUP(lookup_value, range, col_index, [range_lookup])`: 
      * 在 range 中找 lookup_value，如果有找到的話就用 range 內的第 col_index 欄位的結果取代，range_lookup=TRUE 是近似 FALSE (預設) 是 exact match 
      * 如果 range 是別的標籤頁，那就用 **'標籤頁'!範圍**
  * `COUNT(range)`: 計算在 range 內有多少個數值
  * `COUNTA(range)`: 計算在 range 內有多少個非空白的儲存格
  * `COUNTBLANK(range)`: 計算在 range 內有多少個空白的儲存格
    * 注意 " " 會被當成有個空白字串所以不會被列入計算
  * `SUM(range)`: 對 range 內的數值求和
  * `MIN(range)`: 找 range 內的最小數
  * `MAX(range)`: 找 range 內的最大數
  * `AVERAGE(range)`: 算 range 的平均值
  * `MEDIAN(range)`: 找 range 內的中位數
  * `IF(condition, true 時要幹嘛, false 時要幹嘛)`: 條件判斷
  * `AND(condition1, condition2, ...)`: 全部的 condition 都是 true 才傳回 TRUE
  * `OR(condition1, condition2, ...)`: 只要有一個 condition 是 true 就傳回 TRUE
  * `UNIQUE(range)`: 列出 unique 的結果
  * `COUNTIF(range, criteria)`: 計算 range 內有多少個和 criteria 相同的
  * `SUMIF(range, criteria, [sum_range])`: 把 range 內滿足 criteria 的列選出來後，依照 sum_range 求和
  * `AVERAGEIF(range, criteria, [average_range])`: 把 range 內滿足 criteria 的列選出來後，依照 average_range 求平均
  * `AVERAGEIFS(range, criteria1, [average_range1], criteria2, [average_range2], ...)`: 很多條件時用這個
