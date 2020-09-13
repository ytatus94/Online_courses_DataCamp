# 13. Introduction to Data Visualization with Python

## 用 matplotlib 畫圖

* 基本使用方式
  ```python
  import matplotlib.pyplot as plt
  plt.plot([x values], [y values], color='顏色', label='圖標') # 產生在記憶體裡
  plt.show() # 真的畫出來
  ```
  * `x`, `y` 的值可以是 list, np_array, pandas series

* 分割子圖
  * 方法一: `plt.axes([xlo, ylo, width, height])`
    * `xlo`, `ylo`, `width`, `height` 介於 0 ~ 1 是依照 Canvas 的比例分割成子圖
    * 左下角的座標由 `xlo`, `ylo` 決定，這兩個是用 canvas 的座標
    * `plt.axes()` 可以用來畫子母圖
  * 方法二: `plt.subplot(m, n, k)`
    * `m` 列 `n` 行個子圖，目前啟用第 `k` 個圖，`k` 是從 1 開始算起

* 設定圖片的一些參數
  ```python
  # 使用某個圖片 style
  plt.style.use('某種 style')
  
  # 設定 x, y 軸的標題和整張圖的標題
  plt.xlabel('x title')
  plt.ylabel('y title')
  plt.title('canvas title')
  
  # 把 xticks 的文字轉角度
  plt.xticks(rotation=角度) 
  
  # 畫格線
  plt.grid()
  plt.grid('off') # 把格線關掉
  
  # 使用雙 y 軸
  plt.twinx()
  # 在此行程式碼之前的圖看左邊的 y 軸，在此行程式碼之後的圖看右邊的 y 軸
  
  # 設定 x, y 軸的上下限
  plt.xlim([xmin, xmax])
  plt.ylim([ymin, ymax])
  plt.axis([xmin, xmax, ymin, ymax]) # 同時設定 x 和 y 軸上下限，除了用 list 當參數外，也可以用 tuple 當參數
  
  # 圖片的 legend
  plt.legend(loc='圖標的位置')
  # 位置的選擇有'upper left', 'upper center', 'upper right',
  #           'center left', 'center', 'center right',
  #           'lower left', 'lower center', 'low right',
  #           'best', right'
  
  # 在圖片上加上文字說明和箭頭
  plt.annotate(s='要顯示的字',
               xytext=(文字的 x, 文字的 y),
               xy=(箭頭位置的 x, 箭頭位置的 y),
               arrowprops=dict(facecolor='black'))
  # arrowprops 表示 arrow properties 用字典來指定箭頭的特性
  
  # 自動調整 subplots 之間的間距
  plt.tight_layout()
  
  # 把圖片存檔
  plt.savefig('輸出檔案')
  # 支援 pdf, png, jpg 格式
  ```

* 列出所有可用的圖片 style: `print(plt.style.available)`

- plt.axis() 也可以用參數 'off', 'equal', 'square', 'tight'
- axis('off'): turn off axis lines, labels
axis('equal'): equal scaling on x, y axes
axis('square'): forces square plot
axis('tight'): sets xlim(), ylim() to show all data
- 
- 

-
- 
- cs_max = computer_science.max()  和 yr_max = year[computer_science.argmax()] 中的 computer_science 是 numpy.ndarray，可以呼叫 numpy.ndarray 方法 max() 傳回最大值，argmax() 傳回最大值對應的 index
- np.linspace(起點，終點，幾個點) 在起點到終點之間等間隔的產生幾個點，包含終點，產生的是 numpy.ndarray
- X, Y = np.meshgrid(u, v) u, v 都是一維 numpy.ndarray，X, Y 都是二維 numpy.ndarray，若 u 的元素有 n 個 v 的元素有 m 個，則 X, Y 的元素有 m x n 個，實際上 X 是由 m 列個 u 組成，Y 是由 v 轉置後由 n 行組成
- 例如：u = array([-2., -1., 0., 1., 2.]), v = array([-1., 0., 1.]),
則 X = array([[-2., -1., 0., 1., 2.],
                      [-2., -1., 0., 1., 2.],
                      [-2., -1., 0., 1., 2.]])
Y = array([[-1., -1., -1., -1., -1.],
                 [ 0., 0., 0., 0., 0.],
                 [ 1., 1., 1., 1., 1.]])
- plt.pcolor(Z, cmap='blue') pcolor表示 pseudo-color，這圖畫出來上邊跟右側會有留白
- 若再加上 plt.axis('tight') 就不會有留白
- 會留白是因為依照 Z 的值，自動選定 x, y 軸的上限畫圖
若使用 plt.pcolor(X, Y, Z) 也不會有留白，此時 x, y 軸刻度照 X, Y 來畫。
- cmap='jet', 'coolwarm', 'magma', 'viridis', 'greens', 'blues', 'reds', 'purples', 'summer', 'autumn', 'winter', 'spring'
- plt.colorbar() 會在右邊畫一條顏色軸，表示不同顏色代表的意義
- plt.contour(Z, n) n 指定畫幾條 contours
- plt.contour(X, Y, Z, 30) 有 X, Y 則軸就照其刻度
- plt.contourf(Z, 30) 用 contourf() 表示填滿顏色
- plt.hist(x, bins=50)
plt.hist2d(x, y, bins=(20, 30), range=((x下限, x上限), (y下限, y上限))) 2 維的 histogram
- plt.hexbin(x, y, gridsize=(20, 30), extent=(x下限, x上限, y下限, y上限)) 蜂巢形狀的圖
- img = plt.imread('image.jpg') 讀入影像檔，img 是一個 3 維的 numpy.ndarray (x 維度, y 維度, 3)
- intensity = img.sum(axis=2) 求 R G B channel 的和做為強度
- plt.imshow(img, cmap='gray', extent=(x下限, x上限, y下限, y上限), aspect=0.5) aspect 是圖的 height/width 的比例，預設是 aspect='auto'

- 低對比度的圖可以藉由 rescale intensity 來改變
例如：rescaled_image = 256*(image - image.min()) / (image.max() - image.min())




## 用 seaborn 畫圖

* seaborn 是建立在 matplotlib 之上的，所以兩個都要 import
  ```python
  import seaborn as sns
  ```

* 畫 linear regression
  ```python
  sns.lmplot(x='欄位名', y='欄位名', data=df, hue='欄位名', palette='Set1')
  ```
  * `hue` 指定要依照哪個欄位分類，此處的欄位的值是 categorical 的，相同的值就用同樣的顏色
  * `palette` 指定分類的顏色
  * 用 `hue` 是畫在同一張圖上，若是用 `row='欄位名'` 或 `col='欄位名'` 則依照欄位的 categorical 的值，分成 subplot 畫出來，一個 category 畫一個圖，這邊的 `row` 和 `col` 是說在 subplot 上要畫成 row 或 column

* 畫高階 regression
  ```python
  sns.regplot(x='欄位名', y='欄位名', data=df, color='顏色', scatter=None, label='order 2', order=2)
  ```
  * `lmplot()` 只是 `regplot(order=1)` 的特例
  * 階數由 `order` 指定
  * `scatter=None` 表示不要畫數據點，只畫 regression 的線
* 畫 residual 圖
  ```python
  sns.residplot(x='欄位名', y='欄位名', data=df, color='顏色')
  ```
  * residual 就是每個點和 linear regression 的線的距離

* 畫 uni variable 的圖
  ```python
  sns.stripplot(x='欄位名', y='欄位名', data=df, jitter=True, size=3)
  ```
  * `stripplot()` 是把所有數據點畫在一條線上，如果數據值相同，點就疊再一起了
  * 用 `jitter=True` 來讓數據點散開來
  * 用 `size` 指定數據點的大小
  
  ```python
  sns.swarmplot(x='欄位名', y='欄位名', data=df, hue='欄位名', orient='h')
  ```
  * `swarmplot()` 是畫像是樹枝展開一樣的圖，相同的數據點就往左右展開而不會疊再一起
  * `orient='h'` 則是畫成水平的圖
  
  ```python
  sns.violinplot(x='欄位名', y='欄位名', data=df, inner=None, color='lightgray')
  ```
  * `violinplot()` 密度越高的地方越胖
  * `inner=None` 表示不要有 inner annotation
  * 若是先畫 violin plot 再畫 strip plot 那 `stripplot()` 的點會疊在 `violinplot()` 圖上面

* 畫 bivariate 的圖
  ```python
  sns.jointplot(x='hp', y='mpg', data=auto, kind='hex')
  ```
  * `joinplot()` 是中間畫散射圖，上面跟右側放 histogram
  * 可以藉由指定 `kind` 來畫不同的圖
    ```python
    kind='scatter' uses a scatter plot of the data points
    kind='reg' uses a regression plot (default order 1)
    kind='resid' uses a residual plot
    kind='kde' uses a kernel density estimate of the joint distribution
    kind='hex' uses a hexbin plot of the joint distribution
    ```
  * 會計算 Pearson coefficient 和 p-value 的值，並放在圖上
  
  ```python
  sns.pairplot(df, hue='欄位名', kind='reg')
  ```
  * `pairplot()` 把 df 中所有數值欄位都拿來當 `x`, `y` 畫圖
  * 用 `hue` 來區分不同類別的顏色
  * 用 `kind` 來畫不同的圖
  
  ```python
  sns.heatmap(cov_matrix)
  ```
  * `heatmap()` 畫 covariant matrix 所以參數是放 covariant matrix





- 
-  

-  
-  
- 
- time series 可以切片做圖，time series moving window 也可以畫 mean, median, std 等等
- pixels = image.flatten() 把讀入的 image 從 2D numpy array (m x n 個元素) 轉成 1D numpy array (m*n 個元素)
- 
- plt.hist(normed=True, cumulative=True) 表示畫 cumulative distribution function，就是 hist 累積的面積，注意要歸一
- histogram equalization 就是把圖片的像素強度重新計算的一個過程，使得圖片更銳利，對比更明顯，轉換後的圖片的 cdf 會是呈現一條斜線分佈
cdf, bins, patches = plt.hist(pixels, bins=256, range=(0,256), normed=True, cumulative=True) 先取得 cdf
- new_pixels = np.interp(pixels, bins[:-1], cdf*255) 把舊的 pixels 依照 cdf 展開成為新的 pixels
- new_image = new_pixels.reshape(image.shape) 新的 pixels 是一維的，要轉回 2D numpy array，行列的數目要和原本的圖片一樣
- # Extract 2-D arrays of the RGB channels: red, blue, green
red, green, blue = image[:,:,0], image[:,:,1], image[:,:,2]
# Flatten the 2-D arrays of the RGB channels into 1-D
red_pixels = red.flatten()
blue_pixels = blue.flatten()
green_pixels = green.flatten()
