import numpy as np  
from keras.models import Sequential
from keras.datasets import mnist#資料集
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils# 用來後續將 label 標籤轉為 one-hot-encoding  
from matplotlib import pyplot as plt

# 載入 MNIST 資料庫的訓練資料，並自動分為『訓練組』及『測試組』
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# 建立簡單的線性執行的模型(從model = Sequential()~model.compile)
model = Sequential()
# Add Input layer, 隱藏層(hidden layer) 有 256個輸出變數，此處有兩個隱藏層
"""Dense參數
units: 輸出矩陣的維數，愈大表示分類更細，擬合度愈高，雖然準確率提高，但也要防止過度擬合(Overfit)
- activation: 使用的 Activation function，若未設定，即簡化為 y = x * W + b
- use_bias: 是否使用偏差項(Bias)，若未設定或為 False，即簡化為 y = g(x * W)
- kernel_initializer: 權重(W)的初始值
- bias_initializer: 偏差項(Bias)的初始值
"""
model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu')) 
# Add output layer
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))
#model.summary()#模型的摘要信息，其中包括每一层的名称、输出形状和参数数量

# 編譯: 選擇損失函數、優化方法及成效衡量方式
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

# 將 training 的 label 進行 one-hot encoding，例如數字 7 經過 One-hot encoding 轉換後是 0000001000，即第7個值為 1
y_TrainOneHot = np_utils.to_categorical(y_train) 
y_TestOneHot = np_utils.to_categorical(y_test) 
#print("y_TrainOneHot:")
print(y_TrainOneHot)

"""
該程式使用了簡單的全連接神經網絡（MLP）模型，
這種模型期望輸入數據是一個平坦的二維數組
"""
# 將 training 的 input 資料3維(60000,28,28)轉為2維(60000,28*28(input_dim已設定為784))
#.astype('float32') 將數據類型轉換為浮點數型別，增加計算精度
X_train_2D = X_train.reshape(60000, 28*28).astype('float32')  
X_test_2D = X_test.reshape(10000, 28*28).astype('float32')  

"""
正規化是將數據縮放到一個範圍，通常是 [0, 1] 或 [-1, 1]，幫助模型更好地學習
在這裡，數據被除以 255，這是因為圖像像素的範圍通常在 0 到 255 之間。
這樣做可以將像素值縮放到 [0, 1] 的範圍。
"""
x_Train_norm = X_train_2D/255
x_Test_norm = X_test_2D/255

# 進行訓練, 訓練過程會存在 train_history 變數中
train_history = model.fit(x=x_Train_norm, y=y_TrainOneHot, validation_split=0.2, epochs=10, batch_size=800, verbose=2)  

# 顯示訓練成果(分數)
scores = model.evaluate(x_Test_norm, y_TestOneHot)   
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))  

# 預測(prediction)，選擇了測試數據中的前 10 個樣本
X = x_Test_norm[0:15,:]

#predictions = model.predict_classes(X)#原程式
predictions = np.argmax(model.predict(X), axis=-1)
# get prediction result
print("預測結果: ",predictions)

"""
# 顯示 第一筆訓練資料的圖形，確認是否正確
plt.imshow(X_test[3])
plt.show() 
"""

"""
plt.plot(train_history.history['loss'])  
plt.plot(train_history.history['val_loss'])  
plt.title('Train History')  
plt.ylabel('loss')  
plt.xlabel('Epoch')  
plt.legend(['loss', 'val_loss'], loc='upper left')  
plt.show() 
"""

"""
#第一種儲存方式
#將結構存到 model.config 檔案，檔案為JSON或YAML格式
from keras.models import model_from_json
json_string = model.to_json() 
with open(r"D:\0902\神經網路100張圖\1_模型儲存\model.config", "w") as text_file:    
    text_file.write(json_string)

#將權重存到 model.weight 檔案
model.save_weights(r"D:\0902\神經網路100張圖\1_模型儲存\model.weight")
"""

"""
#第二種儲存方式
#儲存結構與權重，檔案的類別為HDF5
from keras.models import load_model
model.save(r"D:\0902\神經網路100張圖\1_模型儲存\model.h5")  # creates a HDF5 file 'model.h5'
"""