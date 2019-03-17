21. Deep Learning in Python

- Forward propagation: 每個節點上算 input 和 weight 的內積
input data 是一個 np.array()，weight 是一個字典包含每個節點上的權重
節點上的內積 = (input_data * weight).sum()
- activation function: 每個節點要用輸入放到 activation function 中計算輸出
用來捕捉非線性的性質
- ReLU: rectified linear activation function:
- relu(x) = 0 if x < 0, relu(x) = x if x > 0
def relu(x):
    return max(x, 0)

loss function 越小表示 model 越好
from sklearn.metrics import mean_squared_error
Gradient descent: 若斜率 >0，往斜率的反方向走會得到較小的值
新的 weight = 舊的 weight - learning rate * slope of weight
weight 的斜率 (就是 gradient) = 算 loss function 的斜率 (就是 error) * 餵入節點的值 *算 activation function 的斜率
When plotting the mean-squared error loss function against predictions, the slope is 2 * x * (y-xb), or 2 * input_data * error
back propagation 就是從 output 往回推 weights，估計 loss function 的斜率
Keras 的四個步驟：建立結構 (多少層，每一層多少節點，activation function 長怎樣)，編譯，fit，預測
用 Keras 回歸：
# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]  有幾個欄位就是表示有幾個 feature

# Set up the model: model
model = Sequential()

# Add the first layer 一層一層加
model.add(Dense(50, activation='relu', input_shape=(n_cols,))) Dense 是某一種 layer，用 50 節點，指明 activation function 用哪個，還要指名輸入層有幾個節點(就是幾個 feature)，然後逗號後沒有指定表示每一列資料都用

# Add the second layer
model.add(Dense(32, activation='relu'))

# Add the output layer
model.add(Dense(1)) 最後一層是輸出層，所以只有一個節點
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error') 至少要指名用哪個 optimizer 和 loss function，Adam 是最常見的optimizer 而 mean squared error 是 regression 中最常見的 loss function
# Fit the model
model.fit(predictors, target)
用 Keras 分類
用 categorical_crossentropy 當 loss function
compile() 要加入 metrics = [‘accuracy’]
因為是分類，要把 target 變成 categorical 的
輸出層是兩層因為有兩種outcome，且要加入 activation='softmax'
範例：
# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

# Convert the target to categorical: target
target = to_categorical(df.survived)

# Set up the model
model = Sequential()

# Add the first layer
model.add(Dense(32, activation='relu', input_shape=(n_cols,)))

# Add the output layer
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(predictors, target)

fit 完之後可以 save, reload, 以後可用來 predict
 In [1]: from keras.models import load_model
In [2]: model.save('model_file.h5')
In [3]: my_model = load_model('my_model.h5')
In [4]: predictions = my_model.predict(data_to_predict_with)

from keras.optimizers import SGD
my_optimizer = SGD(lr=0.1)

model.fit(predictors, target, validation_split=0.3)

from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=2)
model.fit(predictors, target, validation_split=0.3, epochs=30, callbacks=[early_stopping_monitor])
