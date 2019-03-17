16. Statistical Thinking in Python (Part 2)

- slop, intercept = np.polyfit(x, y, 1) 用 numpy 做 linear regression, 第三個參數是說使用的是幾階多項式
- 只是算斜率和截距，畫圖仍然要自己畫
- RSS: residual sum of squares
- np.empty_like(array) 依照 array 的樣子建立一個新的空陣列，新的空陣列和 array 有相同的維度
- 和 np.empty() 有點差別
- Resample an array: 就是從原先的 array 中隨機挑出元素放到新的 array 中
- Bootstrap sample:  resampled array of the data
- bs_sample = np.random.choice(array, size=n) 從 array 中隨機挑出 n 個元素來
- If we have a data set with n repeated measurements, a bootstrap sample is an array of length n that was drawn from the original data with replacement.
- Bootstrap replicate: A statistic computed from a resampled array
只是一個由 bootstrap sample 計算出來的統計數值，
Bootstrap replicate is a single value of a statistic computed from a bootstrap sample.

- 定義 Bootstrap replicate 函數：
def bootstrap_replicate_1d(data, func):
    return func(np.random.choice(data, size=len(data)))

- 定義一個函數產生很多 Bootstrap replicates：
def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""
-     # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)
    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)
    return bs_replicates
或精簡一點的寫法：
def draw_bs_reps(data, func, size=1):
    return np.array([bootstrap_replicate_1d(data, func) for _ in range(size)])
- SEM (standard error of the mean) is given by the standard deviation of the data divided by the square root of the number of data points. I.e., for a data set, sem = np.std(data) / np.sqrt(len(data))
- Bootstrap replicates 的 mean 的分佈是 Normal distribution
- Confidence interval of a statistic:
If we repeated measurements over and over again, p% of the observed values would lie within the p% confidence interval.
np.percentile(bs_replicates, [2.5, 97.5]) 95% confidence interval 是介於 2.5% 到 97.5% 之間正好有 95%
- Pairs bootstrap: 當有兩個陣列要做 Bootstrap 而且陣列有相關時，就用 pairs bootstrap 的方式，bootstrap 陣列的 index 再選出兩陣列中 index 對應的元素
範例程式：
def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""
    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))
    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)
    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds)) 用 index 做 bootstrap
-         bs_x, bs_y = x[bs_inds], y[bs_inds] 挑出 index 對應的 x, y 值
-         bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1) 做 linear regression
-         return bs_slope_reps, bs_intercept_reps
- np.concatenate((array1, array2)) 把兩個 array 結合起來，注意要用 tuple
- np.random.permutation(array) 對 array 內的元素重新排列順序
- Test statistic: A single number that can be computed from observed data and from data you simulate under the null hypothesis.
就是某一個統計量，拿這個統計量來比較不同的資料集
- p-value: The probability of obtaining a value of your test statistic that is at least as extreme as what was observed, under the assumption the null hypothesis is true.
the probability of observing a test statistic equally or more extreme than the one you observed, assuming the hypothesis you are testing is true.
NOT the probability that the null hypothesis is true
- Statistical significance: Determined by the smallness of a p-value
- NHST: Null hypothesis significance testing
- permutation replicate is a single value of a statistic computed from a permutation sample
- permutation test: 有兩筆資料 A 和 B，先組合在一起成一筆大的資料 C，然後對 C 內的元素重新排列順序，把前 a 筆資料叫 A'，後 b 筆資料叫 B'，新的樣本叫做 permutaion sample
用 permutation sample 做 test statistic 把結果稱做 perm_replicates
- p 值：p = np.sum(perm_replicates 和觀察到的做比較) / len(perm_replicates)
- one-sample bootstrap hypothesis test 比較兩筆資料 A 和 B，但因為缺乏其中一筆的數據集，所以只能比較數值而無法比較分佈
- 假設只有 A 的數據集，B 只有 test statistic 的結果（例如只有 B 的平均值)
可以把 A 的數據集平移到 B 上，例如 A - np.mean(A) + B 的 mean 這樣就是新的數據集，有和 B 相同的 mean
用新的數據集求出 bs_replicates
p 值：p = np.sum(bs_replicates 和用 A 的 test statistic 做比較) / len(bs_replicates)
- permutation test 比較精確，但是比較不靈活，因為要有相同的分佈才能用
