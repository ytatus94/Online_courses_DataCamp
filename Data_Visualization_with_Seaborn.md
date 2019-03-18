* 三種畫圖的方式
```python
fig, ax = plt.subplots() # subplots() 其實會回傳三個東西，因為後面要用到 ax 所以有加上等號左邊，不然可以不用加
ax.hist( df['欄位名'] ) # 用 matplotlib 畫圖

df['欄位名'].plot.hist() # 用 pandas 畫圖

sns.distplot( df['欄位名'] ) # 用 seaborn 畫圖，distplot() 畫的是 histogram + Gaussian kernel density estimate (KDE)

plt.show() # 把圖畫出來
plt.clf() # 清除前一張圖
```
  * 用 pandas 畫 histogram 時沒有 x label，預設的 bin size 較寬，y 軸就是次數的累積
  * 用 seaborn 畫 `distplot()` 時有 x label，預設的 bin size 較窄，y 軸是歸一過的
* `distplot()`
```python
sns.distplot(df['欄位名'], hist=False, bins=10, kde=False, rug=True, kde_kws={'shade':True})

sns.distplot(df['欄位名'], ax=ax0) # 第一張圖畫在 ax0
sns.distplot(df.query('欄位判斷式')['欄位名'], ax=ax1) # 第二張圖畫在 ax1
```
畫 `df['欄位名']`，可以選擇要不要畫 hist/kde/rug 預設是會畫 hist 和 kde 疊再一起
  * `kde_kws={'shade':True}` 表示加上 kde keyword 使得 kde 曲線內加上陰影(顏色)
  * `hist_kws={'range':[上限, 下限]}` 可以設定圖中 histogram 的上下限
* `regplot()` 和 `lmplot()`
  * 兩者都是畫 scatter plot 再加上回歸曲線
  * `regplot()` 是比較低階的畫圖，圖的四周有邊框，圖形較扁
    ```python
    sns.regplot(x="欄位名", y="欄位名", data=df, marker='+', color='g', order=n, x_jitter=.1, x_estimator=np.mean, x_bins=4, fit_reg=False)
    ```
    * `order=n` 指定用 n 階多項式
    * `x_jitter=.1`
    * `x_estimator=np.mean`：有太多點時，可以用 mean 在圖上來表示
    * `x_bins=4`：把資料切成 4 個 bins，但是在 fit 時還是用原始 unbinned 的資料在 fit
    * `fit_reg=False`：不要畫 regression line
  * `lmplot()` 是比較高階的畫圖，只有 x 和 y 軸有框，圖形較方型
    ```python
    sns.lmplot(x="欄位名", y="欄位名", data=df, hue="欄位名", row="欄位名", col="欄位名")
    ```
    * 通常放在 hue, row, col 的欄位，會有幾個不同的值，就可以分門別類畫圖
      * `hue="欄位名"`：照 hue 指定的欄位用不同顏色畫在同一張圖
      * `row="欄位名"`：照 row 指定的欄位，將不同的結果畫在不同的列
      * `col="欄位名"`：照 col 指定的欄位，將不同的結果畫在不同的欄

* 設定畫圖的風格
  * 也會影響到 matplotlib 和 pandas 畫的圖
  * `sns.set()` 預設的風格是 `whitegrid`
  * `sns.set_style(style)` style 可以是 `'white'`, `'dark'`, `'whitegrid'`, `'darkgrid'`, `'ticks'`
  * `sns.despine(left=True)` 可以決定拿掉哪邊 (top, bottom, left, right) 的邊框，例如 `left=True` 是拿掉 y 軸
  * `sns.set(color_codes=True)` 使用 matplotlib 的 color code
  * `sns.distplot(df['欄位名'], color='g')` 用 `color='g'` 指定為綠色，
  * matplotlib 的 color code 有 blue, green, red, cyan, magenta, yellow, black, white
  * `sns.palplot()` 顯示 palette
  * `sns.color_palette()` 傳回 current palette
    ```python
    for p in sns.palettes.SEABORN_PALETTES: # 可以顯示有哪些 palette 例如 deep, muted, pastel, bright, dark, colorblind
        sns.set_palette(p) # 用 set_palette() 設定 palette
        sns.palplot( sns.color_palette() )
        plt.show()
    ```
    * color palette 可以分成下面幾類
      * Circular colors = when the data is not ordered 適合 categorial data 時使用
        * 例如：`sns.palplot( sns.color_palette("Paired" , 12) )` 12 表示 12 個顏色
      * Sequential colors = when the data has a consistent range from high to low
        * 例如：`sns.palplot( sns.color_palette("Blues" , 12) )` 12 個不同深淺的藍色
      * Diverging colors = when both the low and high values are interesting
        * 例如：`sns.palplot( sns.color_palette("BrBG" , 12) )`
      * https://seaborn.pydata.org/tutorial/color_palettes.html 可以查可用的 color palette

* 用 matplotlib 的 axes 來自定圖
  * 例如：
  ```python
  fig, ax = plt.subplots() # 要用到 ax 所以等號左邊要加上 fig, ax
  sns.distplot(df['欄位名'], ax=ax) # 指明用 ax 當軸
  ax.set(xlabel="X 軸 title", ylabel="Y 軸 title", xlim=(X 軸下限, X 軸上限), title="圖的 title")
  ```
  * 例如：
  ```python
  fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(7,4)) # 有一列兩欄，共用 y 軸，圖的大小是 7 像素 x 4 像素，因為會有兩張圖，所以等號左邊要有兩個 axes (ax0 和 ax1)
  ```
* Axes
```python
ax1.set( xlabel="X 軸 title" , xlim=(X 軸下限, X 軸上限) )
ax1.axvline(x=20000, label='My Budget' , linestyle='--', linewidth=2) # 在 x = 20000 處畫一條垂直虛線，並加上標籤表明虛線是代表什麼
ax1.legend()
```
* Categorical Plots:
  * stripplot, swarmplot 一類
    * 例如：
    ```python
    sns.stripplot(data=df, x="欄位名", y="欄位名", jitter=True)
    sns.swarmplotplot(data=df, x="欄位名", y="欄位名")
    ```
  * boxplot, violinplot, lvplot 一類
    * 例如：
    ```python
    sns.boxplot(data=df, x="欄位名", y="欄位名")
    sns.violinplot(data=df, x="欄位名", y="欄位名", palette='husl')
    sns.lvplot(data=df, x="欄位名", y="欄位名", palette='Paired', hue='欄位名') # 類似 boxplot + violinplot 的結合
    ```
  * barplot, pointplot, countplot 一類
    * 例如：
    ```python
    sns.barplot(data=df, x="欄位名", y="欄位名", hue="欄位名")
    sns.pointplot(data=df, x="欄位名", y="欄位名", hue="欄位名", capsize=.1)
    sns.countplot(data=df, y="欄位名", hue="欄位名")
    ```
* `residplot()`
  * A residual plot is useful for evaluating the fit of a model
  ```python
  sns.residplot(data=df, x='欄位名', y='欄位名', order=2, color='g')
  ```
* 用 crosstab 產生一個表格：
```python
pd.crosstab(df["欄位名"]當成列索引, df["欄位名"] 當成欄, values=df["欄位名"], aggfunc='mean').round(0) # 最後加上 round() 只是要做四捨五入
```
* `heatmap()`
```python
sns.heatmap(df_crosstab, annot=True, fmt="d", cmap="YlGnBu", cbar=False, linewidths=.5, center=df_crosstab.loc[9, 6])
```
  * `fmd='d'`  用整數表示
  * `cmap='YlGnBu'` 用黃綠藍三色
  * `cbar=False` 關掉圖右邊的 color bar
  * `center=df_crosstab.loc[9, 6]` 用指定的欄位當圖表顏色分布的中間值
  * 可以把 correlation matrix 餵入 `sns.heatmap( df.corr() )`
* Grid plot 有 `FacetGrid()`, `PairGrid()`, `JointGrid()`
  * 所有的 grid plot 都必須要用 tidy format 的輸入資料，每一列表示一筆數據，每一欄表示一個特徵
  * 所有的 grid plot 都要再多一步 `map()` 的步驟
    * 例如：
    ```python
    g = sns.FacetGrid(df, col="欄位名", col_order=[欄位名的列表], row="欄位名", row_order=[欄位名的列表])
    ```
    依照 row 或 col 指定的欄位畫圖，通常該欄位是 categorical 的，不同的值畫在不同張子圖上，這個是由 `row_order` 或 `col_order` 指定，如果沒指定 order 也可以在 `map()` 中指定
  * `g.map(sns.哪一種圖, '欄位名', order=[欄位名的列表])`
    * 例如：
    ```python
    g = sns.FacetGrid(df, col="欄位名")
    g.map(plt.scatter, '欄位名', '欄位名') # 畫散射圖時，要指定散射圖要用的兩個欄位
    ```
  * 例如：
    ```python
    g = sns.PairGrid(df, vars=[欄位名的列表])
    g = g.map(plt.scatter) # 全部都畫成散射圖，但這樣對角線上的圖變成一條由點組成的斜線，沒意義
    g = g.map_diag(plt.hist) # 可以改成指定對角線上畫 histogram
    g = g.map_offdiag(plt.scatter) # 非對角線上畫散射圖
    ```
  * 例如：
    ```python
    g = sns.JointGrid(data=df, x="欄位名", y="欄位名", xlim=(上限, 下限))
    g.plot(sns.regplot, sns.distplot) # 中間圖是散射圖，上面跟右邊圖是 hist + kde
    g = g.plot_joint(sns.kdeplot) # 中間圖是 kde
    g = g.plot_marginals(sns.kdeplot, shade=True) # 上面跟右邊是顏色填滿的 kde
    g = g.annotate(stats.pearsonr) # 中間圖會標上 pearson coef
    ```
* 都有對應的單步驟的圖
  * `FacetGrid()` 對應到 `factorplot()`, `lmplot()`
  * `PairGrid()` 對應到 `pairplot()`
  * `JointGrid()` 對應到 `jointplot()`
  * 例如：
    ```python
    sns.factorplot(x="欄位名", data=df, row='欄位名', row_order=[欄位名的列表], col="欄位名", col_order=[欄位名的列表], kind='哪一種圖', hue='欄位名')
    sns.lmplot(data=df, x="欄位名", y="欄位名", row='欄位名', row_order=[欄位名的列表], col="欄位名", col_order=[欄位名的列表], fit_reg=False) # 畫散射圖可以指定有或沒有回歸曲線
    ```
  * 例如：
    ```python
    sns.pairplot(df, vars=[欄位名的列表], x_vars=[欄位名的列表], y_vars=[欄位名的列表], kind='reg', diag_kind='hist')
    # 可以用 vars 指定要畫的欄位，或是用 x_vars, y_vars 指明哪個畫 x 哪個畫 y
    
    sns.pairplot(df.query('BEDRMS < 3'), vars=[欄位名的列表], kind='scatter/reg', hue='欄位名', palette='husl', plot_kws={'alpha': 0.5})
    ```
  * 例如：
    ```python
    sns.jointplot(data=df, x="欄位名", y="欄位名", kind='hex/reg/resid' , order=n, , xlim=(上限, 下限))
    ```
* 更複雜的自定圖：
```python
g = (sns.jointplot(x="欄位名", y="欄位名", kind='scatter', xlim=(上限, 下限), marginal_kws=dict(bins=15, rug=True), data=df.query(欄位的條件判斷式)).plot_joint(sns.kdeplot))

g = (sns.jointplot(x="欄位名", y="欄位名", kind='scatter', data=df, marginal_kws=dict(bins=10, rug=True)).plot_joint(sns.kdeplot))
```
