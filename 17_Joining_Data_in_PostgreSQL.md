# 17. Joining Data in PostgreSQL

* `SELECT 欄位 FROM 左表格 INNER JOIN 右表格 ON 左表格.key1 = 右表格.key2` 把左右兩表格依照 key 欄位拼在一起，只會用 key 欄位中兩個表格都存在的資料
  * 當兩格表格有相同名字的欄位時，`SELECT` 要指名是用哪個表格的欄位
  * `INNER JOIN` keeps only the records in both tables
* Self join 就是同一個表格自己跟自己 join
  * 通常會在 `ON` 加上另一個條件句，篩除一模一樣的 row 
* `CASE WHEN 條件判斷 THEN 條件為真的時候如何 ELSE 條件為假的時候如何 END`
  * 就是 SQL 的 if else
  * 通常會在 `END` 之後加上 `AS 某名字`
* `SELECT 欄位 INTO 新表格名字 FROM 表格` 從表格中查詢，把查詢結果存到新表格
* Outer join 有 3 種
  * `LEFT JOIN` 保留 left table 全部欄位，而 right table 如果沒有相同的就會用空值
  * `RIGHT JOIN` 保留 right table 全部欄位，而 left table 如果沒有相同的就會用空值
  * `FULL JOIN` 保留 left table 和 right table 全部欄位
* `CROSS JOIN` creates all possibile combinations of two tables



### Examples
```sql
-- 用 INNER JOIN
SELECT * FROM left_table INNER JOIN right_table ON left_table.id = right_table.id;
SELECT * FROM cities;
SELECT *  FROM cities INNER JOIN countries ON cities.country_code = countries.code;
SELECT cities.name AS city, countries.name AS country, countries.region FROM cities INNER JOIN countries ON cities.country_code = countries.code;
SELECT c.code AS country_code, c.name, e.year, e.inflation_rate FROM countries AS c INNER JOIN economies AS e ON c.code = e.code;
SELECT c.code, c.name, c.region, p.year, p.fertility_rate FROM countries AS c INNER JOIN populations AS p ON c.code = p.country_code;
SELECT c.code, name, region, e.year, fertility_rate, unemployment_rate FROM countries AS c INNER JOIN populations AS p ON c.code = p.country_code INNER JOIN economies AS e ON c.code = e.code;
SELECT c.code, name, region, e.year, fertility_rate, unemployment_rate FROM countries AS c INNER JOIN populations AS p ON c.code = p.country_code INNER JOIN economies AS e ON e.code = c.code AND e.year = p.year;
SELECT c.name AS country, continent, l.name AS language, official FROM countries AS c INNER JOIN languages AS l USING (code);

-- 用 self join
SELECT p1.country_code, p1.size AS size2010, p2.size AS size2015 FROM populations AS p1 INNER JOIN populations AS p2 ON p1.country_code = p2.country_code;
SELECT p1.country_code, p1.size AS size2010, p2.size AS size2015 FROM populations as p1 INNER JOIN populations as p2 ON p1.country_code = p2.country_code AND p1.year = p2.year - 5;
SELECT p1.country_code, p1.size AS size2010, p2.size AS size2015, ((p2.size - p1.size)/p1.size * 100.0) AS growth_perc FROM populations AS p1 INNER JOIN populations AS p2 ON p1.country_code = p2.country_code AND p1.year = p2.year - 5;
       
-- 用 CASE WHEN 條件句
SELECT name, continent, code, surface_area, CASE WHEN surface_area > 2000000 THEN 'large' WHEN surface_area > 350000 THEN 'medium' ELSE 'small' END AS geosize_group FROM countries;
SELECT name, continent, code, surface_area, CASE WHEN surface_area > 2000000 THEN 'large' WHEN surface_area > 350000 THEN 'medium' ELSE 'small' END AS geosize_group INTO countries_plus FROM countries;
SELECT country_code, size, CASE WHEN size > 50000000 THEN 'large' WHEN size > 1000000 THEN 'medium' ELSE 'small' END AS popsize_group FROM populations WHERE year = 2015;
SELECT country_code, size, CASE WHEN size > 50000000 THEN 'large' WHEN size > 1000000 THEN 'medium' ELSE 'small' END AS popsize_group INTO pop_plus FROM populations WHERE year = 2015;
SELECT country_code, size, CASE WHEN size > 50000000 THEN 'large' WHEN size > 1000000 THEN 'medium' ELSE 'small' END AS popsize_group INTO pop_plus FROM populations WHERE year = 2015;

-- 用 LEFT JOIN
SELECT name, continent, geosize_group, popsize_group FROM countries_plus AS c INNER JOIN pop_plus AS p ON c.code = p.country_code ORDER BY geosize_group;
SELECT c1.name AS city, code, c2.name AS country, region, city_proper_pop FROM cities AS c1 INNER JOIN countries AS c2 ON c1.country_code = c2.code ORDER BY code DESC;
SELECT c1.name AS city, code, c2.name AS country, region, city_proper_pop FROM cities AS c1 LEFT JOIN countries AS c2 ON c1.country_code = c2.code ORDER BY code DESC;
SELECT c.name AS country, local_name, l.name AS language, percent FROM countries AS c INNER JOIN languages AS l ON c.code = l.code ORDER BY country DESC;
SELECT c.name AS country, local_name, l.name AS language, percent FROM countries AS c LEFT JOIN languages AS l ON c.code = l.code ORDER BY country DESC;
SELECT name, region, gdp_percapita FROM countries AS c INNER JOIN economies AS e ON c.code = e.code WHERE year = 2010;
SELECT region, AVG(gdp_percapita) AS avg_gdp FROM countries AS c LEFT JOIN economies AS e ON c.code = e.code WHERE year = 2010 GROUP BY region;
SELECT region, AVG(gdp_percapita) AS avg_gdp FROM countries AS c LEFT JOIN economies AS e ON c.code = e.code WHERE year = 2010 GROUP BY region ORDER BY AVG(gdp_percapita) DESC;

-- 用 RIGHT JOIN
SELECT cities.name AS city, urbanarea_pop, countries.name AS country, indep_year, languages.name AS language, percent FROM cities LEFT JOIN countries ON cities.country_code = countries.code LEFT JOIN languages ON countries.code = languages.code ORDER BY city, language;
SELECT cities.name AS city, urbanarea_pop, countries.name AS country, indep_year, languages.name AS language, percent FROM languages RIGHT JOIN countries ON languages.code = countries.code RIGHT JOIN cities ON countries.code = cities.country_code ORDER BY city, language;

-- 用 FULL JOIN
SELECT name AS country, code, region, basic_unit FROM countries FULL JOIN currencies USING (code) WHERE region = 'North America' OR region IS NULL ORDER BY region;
SELECT name AS country, code, region, basic_unit FROM countries LEFT JOIN currencies USING (code) WHERE region = 'North America' OR region IS NULL ORDER BY region;
SELECT name AS country, code, region, basic_unit FROM countries INNER JOIN currencies USING (code) WHERE region = 'North America' OR region IS NULL ORDER BY region;
SELECT countries.name, code, languages.name AS language FROM languages FULL JOIN countries USING (code) WHERE countries.name LIKE 'V%' OR countries.name IS NULL ORDER BY countries.name;
SELECT countries.name, code, languages.name AS language FROM languages LEFT JOIN countries USING (code) WHERE countries.name LIKE 'V%' OR countries.name IS NULL ORDER BY countries.name;
SELECT countries.name, code, languages.name AS language FROM languages INNER JOIN countries USING (code) WHERE countries.name LIKE 'V%' OR countries.name IS NULL ORDER BY countries.name;
SELECT c1.name AS country, region, l.name AS language, basic_unit, frac_unit FROM countries AS c1 FULL JOIN languages AS l USING (code) FULL JOIN currencies AS c2 USING (code) WHERE region LIKE 'M%esia';
       
-- 用 CROSS JOIN
SELECT c.name AS city, l.name AS language FROM cities AS c CROSS JOIN languages AS l WHERE c.name LIKE 'Hyder%'
SELECT c.name AS city, l.name AS language FROM cities AS c INNER JOIN languages AS l ON c.country_code = l.code WHERE c.name LIKE 'Hyder%'
SELECT c.name AS country, region, life_expectancy AS life_exp
FROM countries AS c LEFT JOIN populations AS p ON c.code = p.country_code WHERE year = 2010 ORDER BY life_expectancy LIMIT 5;


```
