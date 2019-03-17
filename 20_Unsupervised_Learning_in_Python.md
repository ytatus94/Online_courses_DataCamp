20. Unsupervised Learning in Python

- 找尋資料中的 pattern

- 監督是學習是有標籤，由 pattern 來預測標籤
非監督是學習是沒有標籤，有這兩種方法 clustering, dimension reduction
- column 是 feature，有幾個 columns 就是幾維度的
row 是 sample
- KMeans：(centroids)
- from sklearn.cluster import KMeans
model = KMeans(n_clusters=3) 需要用 n_clusters 指明要分幾群
- 可以先畫散射圖，如果看到大概可以分三群 n_clusters 就用 3
- model.fit(原始輸入樣本) 用輸入樣本來判斷標籤，並且會自動記住，會計算 inertia_ 找最小的 inertia_ 成為一群
- labels = model.predict(新的樣本) 預測新樣本的標籤
- 也可以用 labels = model.fit_predict(samples) 這其實只是先 fit() 再 predict() 組合起來
- centroids = model.cluster_centers_ 計算樣本分群後的每個群中心座標
- 有 3 群的話就有三個群中心座標點
model.inertia_ 求 inertia
- pd.crosstab(df['欄位1'], df['欄位2']) 做 cross-table
- Standardization:
當樣本的 variance 大的時候，表示比較散開，這樣並不適合用 KMeans 來做分群
可以先將樣本作 StandardScalar 使得所有樣本的 variance 都差不多 (mean = 0, variance = 1)
- from sklearn.preprocessing import StandardScaler
scalar = StandardScalar()
scalar.fit(樣本)
sample_scaled = scalar.transform(樣本)

- 然後用 sample_scaled 和 KMeans 來分群 fit(), predice()，不過這樣兩個步驟可以用 pipeline 連接起來：
# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# Create scaler: scaler
scaler = StandardScaler()
# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4)
# Create pipeline: pipeline
pipeline = make_pipeline(scaler, kmeans) 要照順序放，先做 Standardization 在做 KMeans 分群
- # Fit the pipeline to samples
pipeline.fit(samples) 會先 fit StandardScalar 再 fit KMeans
- # Calculate the cluster labels: labels
labels = pipeline.predict(samples)

- 有 Normalizer 可以使用，和 StandardScalar 不同的是 StandardScalar 對欄位做標準化，Normalizer 對列做 rescale
# Import Normalizer
from sklearn.preprocessing import Normalizer
# Create a normalizer: normalizer
normalizer = Normalizer()
df.sort_values(by='欄位') 依照欄位的值來排序
- linkage() function performs hierarchical clustering on an array of samples.
In complete linkage, the distance between clusters is the distance between the furthest points of the clusters.
- In single linkage, the distance between clusters is the distance between the closest points of the clusters.
- from scipy.cluster.hierarchy import linkage, dendrogram
mergings = linkage(samples, method='complete') 用不同的 method 來合併，結果會不同
# Plot the dendrogram, using varieties as labels
dendrogram(mergings, labels=varieties, leaf_rotation=90, leaf_font_size=6,)
from sklearn.preprocessing import normalize
normalized_movements = normalize(樣本)
from scipy.cluster.hierarchy import fcluster
labels = fcluster(mergings, 6, criterion='distance') fcluster() function to extract the cluster labels for this intermediate clustering
- TSNE：把高維度 map 到 2D or 3D

from sklearn.manifold import TSNE
# Create a TSNE instance: model
model = TSNE(learning_rate=200) 學習率一般用 50 到 200
# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(samples) 只有 fit_transform() 沒有個別的 fit() 和 transform()
from scipy.stats import pearsonr
correlation, pvalue = pearsonr(width, length)
PCA (principal components analysis) 把資料移到 (rotate + shift) 新的軸上 (新的軸叫做 PCA feature) 使得 mean = 0 並且移除 correlation，所以 PCA 後的資料 Pearson correlation = 0，PCA 後的資料並沒有損失資訊
principal components = direction of variance
# Import PCA
from sklearn.decomposition import PCA
# Create PCA instance: model
model = PCA() 可以放入參數 n_components=2 強制使用 intrinsic dimension = 2 的，PCA 會挑出兩個有最大 PCA feature variance 的來使用
# Apply the fit_transform method of model to grains: pca_features
pca_features = model.fit_transform(grains) 也有 fit() 和 transform()，fit() 只是先計算該如何 rotate + shift，transform() 才是真的變過去
The first principal component of the data is the direction in which the data varies the most.
mean = model.mean_ 獲得 PCA 之後的平均值
first_pc = model.components_[0,:]  model.components_ 每一列表示從 mean 的位移
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)
intrinsic dimension: 有最大 variance 的 PCA feature 數目
有較小的 PCA feature variance 就當作是 noise，大的 PCA feature variance 才當作是有用的資訊
features = range(pca.n_components_) pca.n_components_ 算有幾個成分
plt.bar(features, pca.explained_variance_)
# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer()
# Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(documents) TfidfVectorizer 把一個文件 list 轉換成 word frequency array
# Print result of toarray() method
print(csr_mat.toarray())
# Get the words: words
words = tfidf.get_feature_names()
# Print words
print(words)
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=50)
from sklearn.decomposition import NMF
model = NMF(n_components=6) NMF 一定要指定 n_components
