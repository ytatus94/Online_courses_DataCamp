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
* Aggregate functions: `AVG()`, `SUM()`, `MAX()`, `MIN()`
  * 如果不用 alias 的話，aggregate function 產生的新欄位名字就是 function 的名字
* `ORDER BY` 預設是 ascending order, 如果要用 descending order 要加上 `DESC`
  * 如果是 text 的資料就依照 A-Z 來排序
  * 可以依照好幾個欄位來排序
* `GROUP BY` 通常和 aggregate function 一起用
  * `GROUP BY` 放在 `FROM` 後面，放在 `ORDER BY` 前面
  * 如果 `SELECT` 一個不是 `GROUP BY` 的欄位，而沒對該欄位做 aggregate function 的計算，那 SQL 會報錯
* `--` 註解單行 `/* */` 註解一個範圍
* `WHERE` 子句不可以用 aggregate function，要改成用 `HAVING` 子句
  * 若有 `GROUP BY` 則 `HAVING` 子句放在 `GROUP BY` 後面
* `LIMIT` 可以限制輸出的 rows 數目 
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
-- 可以用 AND 和 OR 來增加條件句，記得有些時候要加上括號，順序才不會出錯
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

-- 用 BETWEEN 來篩選出指定的範圍
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

-- 用 aggregate functions
SELECT SUM(duration) FROM films;
SELECT AVG(duration) FROM films;
SELECT MIN(duration) FROM films;
SELECT MAX(duration) FROM films;
SELECT SUM(gross) FROM films;
SELECT AVG(gross) FROM films;
SELECT MIN(gross) FROM films;
SELECT MAX(gross) FROM films;
SELECT SUM(gross) FROM films WHERE release_year >= 2000;
SELECT AVG(gross) FROM films WHERE title LIKE 'A%';
SELECT MIN(gross) FROM films WHERE release_year = 1994;
SELECT MAX(gross) FROM films WHERE release_year BETWEEN 2000 AND 2012;

-- 用 alias
SELECT title, (gross - budget) AS net_profit FROM films;
SELECT title, (duration / 60.0) AS duration_hours FROM films;

-- 用 aggregate function 和 alias
-- SQL 除法如果分子分母都是整數，則商也會是整數，因此要加上小數點
SELECT AVG(duration) / 60.0 AS avg_duration_hours FROM films;
SELECT (COUNT(deathdate) * 100.0 / COUNT(*)) AS percentage_dead FROM people;
SELECT MAX(release_year) - MIN(release_year) AS difference FROM films;
SELECT (MAX(release_year) - MIN(release_year))/ 10 AS number_of_decades FROM films;

-- 用 ORDER BY 來排序
SELECT name FROM people ORDER BY name
SELECT name FROM people ORDER BY birthdate
SELECT birthdate, name FROM people ORDER BY birthdate
SELECT title FROM films WHERE release_year IN (2000, 2012) ORDER BY release_year;
SELECT * FROM films WHERE release_year NOT IN (2015) ORDER BY duration;
SELECT title, gross FROM films WHERE title LIKE 'M%' ORDER BY title;
SELECT imdb_score, film_id FROM reviews ORDER BY imdb_score DESC;
SELECT title FROM films ORDER BY title DESC;
SELECT title, duration FROM films ORDER BY duration DESC;
SELECT birthdate, name FROM people ORDER BY birthdate, name;
SELECT release_year, duration, title FROM films ORDER BY release_year, duration;
SELECT certification, release_year, title FROM films ORDER BY certification, release_year;
SELECT name, birthdate FROM people ORDER BY name, birthdate;

-- 用 GROUP BY
SELECT release_year, COUNT(*) FROM films GROUP BY release_year
SELECT release_year, AVG(duration) FROM films GROUP BY release_year
SELECT release_year, MAX(budget) FROM films GROUP BY release_year
SELECT imdb_score, COUNT(*) FROM reviews GROUP BY imdb_score
SELECT release_year, MIN(gross) FROM films GROUP BY release_year;
SELECT language, SUM(gross) FROM films GROUP BY language;
SELECT country, SUM(budget) FROM films GROUP BY country;
SELECT release_year, country, MAX(budget) FROM films GROUP BY release_year, country ORDER BY release_year, country;
SELECT country, release_year, MIN(gross) FROM films GROUP BY release_year, country ORDER BY country, release_year;

-- 用 HAVING 篩選 aggregate function 的結果
SELECT release_year FROM films GROUP BY release_year HAVING COUNT(title) > 200;
SELECT release_year, budget, gross FROM films;
SELECT release_year, budget, gross FROM films WHERE release_year > 1990;
SELECT release_year FROM films WHERE release_year > 1990 GROUP BY release_year;
SELECT release_year, AVG(budget) as avg_budget, AVG(gross) as avg_gross FROM films WHERE release_year > 1990 GROUP BY release_year;
SELECT release_year, AVG(budget) as avg_budget, AVG(gross) as avg_gross FROM films WHERE release_year > 1990 GROUP BY release_year HAVING AVG(budget) > 60000000;
SELECT release_year, AVG(budget) as avg_budget, AVG(gross) as avg_gross FROM films WHERE release_year > 1990 GROUP BY release_year HAVING AVG(budget) > 60000000 ORDER BY AVG(gross) DESC;
SELECT country, AVG(budget) AS avg_budget, AVG(gross) AS avg_gross FROM films GROUP BY country HAVING COUNT(title) > 10 ORDER BY country LIMIT 5

-- 用 JOIN
SELECT title, imdb_score FROM films JOIN reviews ON films.id = reviews.film_id WHERE title = 'To Kill a Mockingbird';
```
