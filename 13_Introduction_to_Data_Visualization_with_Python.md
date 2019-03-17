13. Introduction to Data Visualization with Python

- import matplotlib.pyplot as plt
plt.plot([x values], [y values], color='顏色', label='圖標') 產生在記憶體裡。
- x, y 的值可以是 list, np_array, pandas series。
plt.show() 真的畫出來。

- 分割子圖：
方法一：plt.axes([xlo, ylo, width, height])
- xlo, ylo, width, height 介於 0 ~ 1 是依照 Canvas 的比例分割成子圖
左下角的座標由 xlo, ylo 決定，這兩個是用 canvas 的座標。
plt.axes() 可以用來畫子母圖。
- 方法二：plt.subplot(m, n, k)
- m 列 n 行個子圖，目前啟用第 k 個圖，k 是從 1 開始算起。
- plt.xlabel('x title')，plt.ylabel('y title')，plt.title('canvas title') 標註 x, y 軸和圖的標題
- plt.tight_layout() improve the spacing between subplots
- plt.xlim([xmin, xmax]) 和 plt.ylim([ymin, ymax]) 可個別設定 x 和 y 軸上下限
- plt.axis([xmin, xmax, ymin, ymax]) 則是同時設定，除了用 list 當參數外，也可以用 tuple 當參數。
- plt.axis() 也可以用參數 'off', 'equal', 'square', 'tight'
- axis('off'): turn off axis lines, labels
axis('equal'): equal scaling on x, y axes
axis('square'): forces square plot
axis('tight'): sets xlim(), ylim() to show all data
- plt.legend(loc='圖標的位置') 有 'upper left', 'upper center', 'upper right', 'center left', 'center', 'center right', 'lower left', 'lower center', 'low right', 和 'best', right'
- plt.annotate(s='要顯示的字', xy=(箭頭的 x, 箭頭的 y), xytext=(字的 x, 字的 y), arrowprops=dict(facecolor='black')) arrowprops 表示 arrow properties 用字典來指定特性
- plt.style.use('ggplot') 使用 ggplot 的 style
- 用 print(plt.style.available) 列出所有可用的 style
- plt.savefig('輸出檔案.png') 支援 pdf, png, jpg 格式
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
- import seaborn as sns seaborn 是建立在 matplotlib 之上的，所以兩個都要 import
- sns.lmplot(x='欄位名', y='欄位名', data=df, hue='欄位名', palette='Set1') 畫 linear regression，hue 指定要依照哪個欄位分類，此處的欄位的值是 categorical 的，相同的值就用同樣的顏色，而 palette 指定分類的顏色，用 hue 是畫在同一張圖上，若是用 row='欄位名' 或 col='欄位名' 則依照欄位的 categorical 的值，分成 subplot 畫出來，一個 category 畫一個圖，這邊的 row 和 col 是說在 subplot 上要畫成 row 或 column
- sns.residplot(x='欄位名', y='欄位名', data=df, color='green') 畫 residual 圖，residual 就是每個點和 linear regression 的線的距離
- sns.regplot(x='欄位名', y='欄位名', data=df, color='green', scatter=None, label='order 2', order=2) regplot() 可以畫高階 regression 圖，階數由 order 指定，lmplot() 只是 regplot(order=1) 的特例，scatter=None 表示不要畫數據點，只畫 regression 的線
- sns.stripplot(), sns.swarmplot(), sns.violinplot() 畫的是 uni variable 的圖
- sns.joinplot(), sns.pairplot(), sns.heatmap() 畫的是 bivariate 的圖
- sns.stripplot(x='欄位名', y='欄位名', data=df, jitter=True, size=3) strip plot 是把所有數據點畫在一條線上，如果數據值相同，點就疊再一起了，所以可以用 jitter=True 來讓點散開來，用 size 指定點的大小
- sns.swarmplot(x='欄位名', y='欄位名', data=df, hue='欄位名', orient='h') swarm plot 是畫像是樹枝展開一樣的圖，相同的數據點就往左右展開而不會疊再一起，orient='h' 則是說畫成水平的圖
- sns.violinplot(x='欄位名', y='欄位名', data=df, inner=None, color='lightgray') violin plot 密度越高的地方越胖，inner=None 表示不要有 inner annotation。
- 若是先畫 violin plot 再畫 strip plot 那 strip plot 的點會疊在 violin plot 上面
- sns.jointplot(x='hp', y='mpg', data=auto, kind='hex') join plot 是中間畫散射圖，上面跟右側放 histogram，可以藉由指定 kind 來畫不同的圖，會計算 Pearson coefficient 和 p-value 的值並放在圖上
- kind='scatter' uses a scatter plot of the data points
- kind='reg' uses a regression plot (default order 1)
- kind='resid' uses a residual plot
- kind='kde' uses a kernel density estimate of the joint distribution
- kind='hex' uses a hexbin plot of the joint distribution
- sns.pairplot(df, hue='欄位名', kind='reg') pair plot 把 df 中所有數值欄位都拿來當 x, y 畫圖，可以用 hue 來區分不同類別的顏色，用 kind 來畫不同的圖
- sns.heatmap(cov_matrix) heat map 畫 covariant matrix 所以參數是放 covariant matrix
- plt.xticks(rotation=60) 把 xticks 的文字轉 60 度
- time series 可以切片做圖，time series moving window 也可以畫 mean, median, std 等等
- pixels = image.flatten() 把讀入的 image 從 2D numpy array (m x n 個元素) 轉成 1D numpy array (m*n 個元素)
- plt.grid('off')
- plt.twinx() 雙 y 軸，在此行之前的圖看左邊的 y 軸，在此行之後的圖看右邊的 y 軸
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
