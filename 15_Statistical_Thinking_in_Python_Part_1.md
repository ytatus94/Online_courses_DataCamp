15. Statistical Thinking in Python (Part 1)

- EDA: exploratory data analysis
- plt.xxx() 會回傳三個 array 但是我們不感興趣，可以用底線設成 dummy variables： _ = plt.xxx()
- import seaborn as sns
sns.set() 使用 seaborn 的預設 style
- plt.hist(可以用 data frame 的欄位或是 numpy array)
hist 預設是 10 bins 可以用 bins=XX 來修改數目，一般來說用 number of sample 的平方根來當 bins 數目，XX 可以是數字，指明幾個 bins 也可以是 list 列出 bin edge.

- 畫圖時記得要標上 xlabel 和 ylable 還有單位也要標上去
- plt.margins(0.02) 圖的邊邊和軸保持 0.02 的距離，這樣在軸上的點才比較看得清楚
- Binning bias: 因為選擇的 bin 數目不同，使得同樣的資料畫出來的圖看起來不一樣，因此造成解釋不同。
- Bee swarm plot 又叫做 swarm plot
sns.swarmplot(x='類別欄位', y='數據欄位', data=df) 類別與數據都是 df 的欄位名稱
- ECDF: Empirical cumulative distribution function
- 總是先畫 ECDF 看一下
ECDF 表示有 y軸% 的資料小於等於對應的 X 軸的值
ECDF 的 x 是排序過的， y 是從 1/n 到 1
x = np.sort(data)
y = np.arange(1, n+1) / n 其中 n = len(data) 要用 n+1 是因為最後一項不包含在內

- 定義一個 ECDF 函數
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n + 1) / n
    return x, y

- 畫 ECDF 圖時，plt.plot() 預設是畫線圖，所以要加上 linestyle='none' 來只顯示點
- plt.plot(x, y, marker='.', linestyle='none')
- np.percentile(Numpy array 或 df['欄位'], 多少 percentile)
例如：np.percentile(df['欄位'], [25, 50, 75]) 會算出該欄位的 25%, 50%, 75% percentile 的值

- 把 percentile 畫到 ECDF 上：plt.plot(ptiles_vers, percentiles/100, marker='D', color='red', linestyle='none') 記得 y 軸要除以 100 因為 ECDF 的 y 值是介於 0 與 1

- 介於 25% 到 75% 之間的範圍叫做 IQR
box plot 上下橫槓 (英文叫做 whisker) 是距離 25% 或 75% 1.5 IQR 處，一般把 2 IQR 以外的當成 outliers

- 畫 box plot: sns.boxplot(x='類別欄位', y='數據欄位', data=df)
- covariance = \frac{1}{n} \sum_i (x_i - \bar{x})(y_i - bar{y})
- Pearson correlation coefficient = \rho = covariance / ( (std x)(std y) ) = -1 ~ 1 dimensionless
定義 Pearson correlation coefficient 函數：
- def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x, y)
    # Return entry [0,1]
    return corr_mat[0,1]

- 產生亂數種子 np.random.seed(n)  n 是隨便填的數字，有填數字的話，每次重跑程式的結果才會一樣
- np.empty(100000) 產生一個空的 np_array 裡面有 100,000 個空的數值
- np.random.random() 產生亂數
- Bernoulli trials: An experiment that has two options, "success" (True) and "failure" (False).

- 定義 Bernoulli trials 函數：
n 次試驗，每次成功的機會為 p，傳回 n 次試驗中成功的次數
def perform_bernoulli_trials(n, p):
    """Perform n Bernoulli trials with success probability and return number of successes."""
    # Initialize number of successes: n_success
    n_success = 0
    # Perform trials
    for i in range(n):
        # Choose random number between zero and one: random_number
        random_number = np.random.random()
        # If less than p, it's a success so add one to n_success
        if random_number < p:
            n_success += 1
        return n_success
- PMF: Probability mass function
- A mathematical description of outcomes (其實就是 PDF 不連續時的版本)
- The number r of successes in n Bernoulli trials with probability p of success, is Binomially distributed
- np.random.binomial(n trials, probability, size=重複做幾次)
- Poisson process: The timing of the next event is completely independent of when the previous event happened
Poisson distribution: The number r of arrivals of a Poisson process in a given time interval with average rate of λ arrivals per interval is Poisson distributed.
- Poisson Distribution: Limit of the Binomial distribution for low probability of success and large number of trials.
- np.random.poisson(mean, size=重複做幾次)
- np.random.normal(mean, std, 產生多少個事件)
- np.sum(可以放 Numpy array 的條件判斷)
- The Exponential distribution: The waiting time between arrivals of a Poisson process is Exponentially distributed
