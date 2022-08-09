# 11. Intro to SQL for Data Science

* SQL 的關鍵字是不分大小寫的
* SQL 命令結尾要加上**分號**
* 用 `SELECT *` 表示顯示所有欄位
* 用 `DISTINCT 欄位` 來只顯示該欄位的 unique values
* `WHERE` 可以用來篩選 text 和 numeric values
  * 可使用下面的條件判斷符號
    * `=` equal
    * `<>` not equal **注意!** 不等於是用 `<>` 不是用 `!=` (某些 SQL client 可以用 `!=`，最好還是用 `<>`)
    * `<` less than
    * `>` greater than
    * `<=` less than or equal to
    * `>=` greater than or equal to
  * `WHERE` 放在 `FROM` 後面
  * 篩選 text 時大部分的 SQL client 用單引號把 text 括起來
  * 用 `AND` 來連結不同的條件判斷句
* `BETWEEN` 是 inclusive 所以 `BETWEEN start AND end` 包含開始 start 和結束 end 的數值
* `IS NULL` 和 `IS NOT NULL` 可判斷是否是空值，通常用在 `WHERE` 子句中
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

## Examples
```sql
-- 從表格中選擇欄位
SELECT title FROM films;
SELECT release_year FROM films;
SELECT name FROM people;
SELECT title FROM films;
SELECT title, release_year FROM films;
SELECT title, release_year, country FROM films;
SELECT * FROM films;

-- 用 DISTINCT 顯示欄位中的 unique values
SELECT DISTINCT country FROM films;
SELECT DISTINCT certification FROM films;
SELECT DISTINCT role FROM roles;

-- 用 COUNT() 計算 rows 數目
-- COUNT(*) 計算表格總共有多少 rows
-- COUNT(欄位) 計算該欄位總共有多少 non-missing values
-- COUNT(DISTINCT 欄位) 計算該欄位總共有多少 distinct values
SELECT COUNT(*) FROM reviews;
SELECT COUNT(*) FROM people;
SELECT COUNT(birthdate) FROM people;
SELECT COUNT(DISTINCT birthdate) FROM people;
SELECT COUNT(DISTINCT language) FROM films;
SELECT COUNT(DISTINCT country) FROM films;

-- 用 WHERE 來篩選
SELECT * FROM films WHERE release_year = 2016;
SELECT count(*) FROM films WHERE release_year < 2000;
SELECT title, release_year FROM films WHERE release_year > 2000;
SELECT * FROM films WHERE language = 'French';
SELECT name, birthdate FROM people WHERE birthdate = '1974-11-11';
SELECT count(*) FROM films WHERE language = 'Hindi';
SELECT * FROM films WHERE certification = 'R';
SELECT title, release_year FROM films WHERE language = 'Spanish' AND release_year < 2000;
SELECT * FROM films WHERE language = 'Spanish' AND release_year > 2000;
SELECT * FROM films WHERE language = 'Spanish' AND release_year > 2000 AND release_year < 2010;
SELECT title, release_year FROM films WHERE release_year > 1989 AND release_year < 2000;
SELECT title, release_year FROM films WHERE release_year > 1989 AND release_year < 2000 AND (language = 'French' OR language = 'Spanish');
SELECT title, release_year FROM films WHERE release_year > 1989 AND release_year < 2000 AND (language = 'French' OR language = 'Spanish') AND gross > 2000000;
SELECT title, release_year from films WHERE release_year BETWEEN 1990 AND 2000;
SELECT title, release_year from films WHERE release_year BETWEEN 1990 AND 2000 AND budget > 100000000;
SELECT title, release_year from films WHERE release_year BETWEEN 1990 AND 2000 AND budget > 100000000 AND language = 'Spanish';
SELECT title, release_year from films WHERE release_year BETWEEN 1990 AND 2000 AND budget > 100000000 AND (language = 'Spanish' or language = 'French');

-- 用 IN (項目1, 項目2, ...) 來篩選
SELECT title, release_year FROM films WHERE release_year IN (1990, 2000) AND duration > 120;
SELECT title, language FROM films WHERE language IN ('English', 'Spanish', 'French');
SELECT title, certification FROM films WHERE certification IN ('NC-17', 'R');

-- 用 IS NULL 和 IS NOT NULL 來判斷空值
SELECT name FROM people WHERE deathdate IS NULL;
SELECT title FROM films WHERE budget IS NULL;
SELECT count(*) FROM films WHERE language IS NULL;

-- 用 LIKE 和 NOT LIKE 來匹配
SELECT name FROM people WHERE name LIKE 'B%';
SELECT name FROM people WHERE name LIKE '_r%';
SELECT name FROM people WHERE name NOT LIKE 'A%';
```
