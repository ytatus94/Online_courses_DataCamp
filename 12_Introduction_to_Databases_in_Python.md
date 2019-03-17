12. Introduction to Databases in Python

- 讀取 SQL
from sqlalchemy import create_engine
engine = create_engine('sqlite:///檔案.sqlite') 建立介面，連結到 'driver:///檔案'
- 這種方式可以適用於各種支持的 SQL 不一定只是 sqlite 而已
例如：engine = create_engine('postgresql+psycopg2://student:datacamp@postgresql.csrrinzqubik.us-east-1.rds.amazonaws.com:5432/census') 這是使用 PostgreSQL
- 例如：engine = create_engine('mysql+pymysql://student:datacamp@courses.csrrinzqubik.us-east-1.rds.amazonaws.com:3306/census') 這是使用 MySQL
- engine.table_names() 可以知道 SQL 檔案中所有的表格名字
- connection = engine.connect() 和 SQL 檔案連接起來
- stmt = 'SQL 指令'
result_proxy = connection.execute(stmt) 這邊才是真正執行 SQL 指令
- results = result_proxy.fetchall()
之後就可以使用 results 的結果
例如：first_row = results[0] 只取第一列
first_row.keys() 顯示所有欄位名字，因為第一列就是欄位的名字
first_row.欄位 得到第一列的某欄位的值

- 使用 reflection 可以自動讀取表格，並且依照表格資訊建立 metadata
from sqlalchemy import MetaData, Table
- metadata = MetaData() 依照讀取的資料自動建立 metadata
- 表格 = Table('表格名字', metadata, autoload=True, autoload_with=engine) 通常表格和表格名字是用一樣的
- # Print table metadata
print(repr(表格)) 顯示表格欄位名字與型態
- # Print the column names
print(表格.columns.keys()) 顯示表格欄位名字
- # Print full table metadata
print(repr(metadata.tables['census'])) 其實結果和 print(repr(表格)) 一樣

- 用 pythonic 的方法來讀取 SQL
from sqlalchemy import select
SELECT 敘述：
stmt = select([表格名字]) 選擇表格中的所有欄位，或是使用 stmt = select([表格.columns.欄位1, 表格.columns.欄位2, ...]) 選擇特定的欄位
- WHERE 條件判斷：
stmt = stmt.where(表格.columns.欄位 ==,<=,>=,!=之類的條件判斷)
也可以用 in_(), like(), between()
- 例如：stmt = stmt.where(census.columns.state.in_(states))
- 如果需要用到 and, or, not 時
from sqlalchemy import and_, or_, not_
and_(條件判斷1, 條件判斷2) 或是用 or_(條件判斷1, 條件判斷2)
- 排序 ORDER BY column1, column2, ... ASC|DESC：
stmt = stmt.order_by(表格.columns.欄位1, 表格.columns.欄位2, ...) 可以依照欄位來遞增排序
- 要遞減排序的話就用 desc(要遞減排序的欄位列表) ：
from sqlalchemy import desc
stmt.order_by(表格.columns.欄位1, desc(表格.columns.欄位2)) 欄位1遞增排序，欄位2遞減排序
- 如果需要用到 aggregation function：
from sqlalchemy import func
- 例如：stmt = select([func.count(census.columns.state.distinct())]) 注意 distinct() 加在最後面
- distinct_state_count = connection.execute(stmt).scalar()
- 例如：stmt = select([census.columns.state, func.count(census.columns.age)])
當有用到 aggregation function 時，表格的欄位會變成所用的 func 名字後綴底線，例如 count_，可以用 .label('新欄位名字') 來取代，相當於 SQL 中的 AS
- 例如：pop2008_sum = func.sum(census.columns.pop2008).label('population')
例如：stmt = select([census.columns.state, (census.columns.pop2008-census.columns.pop2000).label('pop_change')]) 這邊第二個參數先做計算，算 pop2008 和 pop2000 的差
群組 GROUP BY：
stmt = stmt.group_by(表格.columns.欄位)
- LIMIT 敘述：
stmt = stmt.limit(幾行輸出)
CASE 敘述：
from sqlalchemy import case
- case([條件判斷, True 時如何], else_ 否則如何)
例如：stmt = select( [ func.sum(case( [ (census.columns.state == 'New York', census.columns.pop2008)], else_=0))])
- 型態轉換範例：
from sqlalchemy import case, cast, Float 引用 cast 和 Float 是為了要做型態轉換
- female_pop2000 = func.sum(case( [ (census.columns.sex == 'F', census.columns.pop2000)], else_=0))
- # Cast an expression to calculate total population in 2000 to Float
total_pop2000 = cast(func.sum(census.columns.pop2000), Float)
# Build a query to calculate the percentage of females in 2000: stmt
stmt = select([female_pop2000 / total_pop2000* 100])
# Execute the query and store the scalar result: percent_female
percent_female = connection.execute(stmt).scalar()
JOIN 兩個表格：
stmt = stmt.select_from(表格1.join(表格2, 依照表格一的哪個欄位 == 依照表格二的哪個欄位))
若是兩個表格有相同欄位，又用此欄位 JOIN 那第二個參數可以省略
例如：stmt = stmt.select_from(census.join(state_fact, census.columns.state == state_fact.columns.name))
- print(stmt) 印出 SQL 命令，可以看命定到底有沒有寫對
- 最後執行 SQL 命令並取得資料：
results = connection.execute(stmt).fetchall() 得表格中的資料
- result = connection.execute(stmt).first() 得表格中的第一筆資料 只取得表格中的第一筆資料
- for result in results: 取得資料做想做的事情
- 也可以不經過 results 而直接使用 for result in connection.execute(stmt):

- 讀取出來的 results 可以直接餵給 Pandas DataFrame 但是欄位名字要自己指定
df = pd.DataFrame(results)
df.columns = results[0].keys() 第一列正好存放的是欄位的名字
- 讀入 DataFrame 後就可以做各種操作

- 新表格名字 = 舊表格名字.alias()
例如：managers = employees.alias() managers 和 employees 是同一個表格，但是現在可以用不同的名字來做操作了，通常都是自己要 join 自己，然後做些條件判斷

- 當表格資料太大時，可以一次只取得一部分，然後用迴圈來讀取全部的資料
# Start a while loop checking for more results
while more_results:
    # Fetch the first 50 results from the ResultProxy: partial_results
-     partial_results = results_proxy.fetchmany(50) 一次只讀50筆資料
-     # if empty list, set more_results to False
-     if partial_results == []: 如果讀到的資料為空列表，表示讀完全部了
-         more_results = False
    # Loop over the fetched records and increment the count for the state
    for row in partial_results: 對讀到的資料做處理
-         loop body
results_proxy.close() 用這個方法記得要手動關閉

- 建立表格
from sqlalchemy import Table, Column, String, Integer, Float, Boolean 把所有要用到的型態都 import
- data = Table(‘表格名字’, metadata,
-                       Column('欄位名字', 欄位型態), ...)
# Use the metadata to create the table
metadata.create_all(engine) 建立表格
- # Print table details
print(repr(data)) 印出來檢查看看
- 建立表格時，可以再 Column() 中加參數 unique=True/False, nullable=True/False, default=預設值

- 插入表格方法一：
from sqlalchemy import insert
stmt = insert(表格).values(欄位1=值1, 欄位2=值2, ...)
- results = connection.execute(stmt)
print(results.rowcount) 總是印出來看看剛剛處理了多少筆資料，以防不小心做錯
- 插入表格方法二：
values_list = [ {欄位1:值1, 欄位2:值2, ...}, {欄位1:值3, 欄位2:值4, ...} ]
- stmt = insert(表格)
- results = connection.execute(stmt, values_list)

- 更新表格：
stmt = update(表格).values(欄位=值)
- stmt = stmt.where(表格.columns.欄位 ==值) 要用 where 決定是哪個要更新
- results = connection.execute(stmt) 這邊才是真的去更新

- 刪除表格全部資料：
from sqlalchemy import delete
stmt = delete(表格)
- results = connection.execute(stmt)

- 刪除表格資料：
delete(表格).where(表格.columns.欄位 ==值) 使用 where 來決定哪筆資料需要刪除

- 刪除表格：
表格.drop(engine)

- 刪除全部表格：
metadata.drop_all(engine)
