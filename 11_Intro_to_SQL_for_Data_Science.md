# 11. Intro to SQL for Data Science

* SQL 的關鍵字是不分大小寫的
* SQL 命令結尾要加上**分號**
* 不等於是用 `<>` 不是用 `!=` (某些客戶端可以用 !=，最好還是用 <>)
* `BETWEEN` 是 inclusive
* `IS NULL` 和 `IS NOT NULL` 可判斷是否是空值
* `LIKE` 和 `NOT LIKE` 加上通配符 `%` (多個字符) 或 `_` (一個字符)
* `WHERE age = 1 OR age = 3 OR age = 5` 可以改成 `WHERE age IN (1, 3, 5)`
* `GROUP BY` 放在 `ORDER BY` 前面
* `--` 註解單行 `/* */` 註解一個範圍
* `WHERE` 子句不可以用 aggregate function，要改成用 `HAVING` 子句
  * 若有 `GROUP BY` 則 `HAVING` 子句放在 `GROUP BY` 後面
* 基本語法:
  * 關鍵字的順序要對！
```SQL
SELECT 欄位1, DISTINCT  欄位2, AVG(欄位3) AS 新名稱, COUNT(DISTINCT 欄位4)
FROM 表格1
JOIN 表格2
ON 表格1.欄位 = 表格2.欄位
WHERE 條件1 AND (條件2 OR 條件3)
GROUP BY 欄位1
HAVING AVG(欄位3)條件
ORDER BY 欄位2
LIMIT 數目
```
