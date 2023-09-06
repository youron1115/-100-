#使用前須將終端機的目錄改變到此程式所在的目錄:"D:\0902\神經網路100張圖"
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import os
import sys

# 計算相似矩陣
def cosine_similarity(ratings):
    sim = ratings.dot(ratings.T)
    if not isinstance(sim, np.ndarray):
        sim = sim.toarray()
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)
    
def main_func(inputNo):
    # 自 images 目錄找出所有 JPEG 檔案    
    y_test=[]#照片名稱
    x_test=[]#包含所有处理过的图像数据
    
    for img_path in os.listdir("images"):
        #print("img_path=",img_path)
        if img_path.endswith(".jpg"):
            img = image.load_img("images/"+img_path, target_size=(224, 224))
            y_test.append(img_path[:-3])#將檔名後三個字元.jpg去掉後當圖案標籤加進y_test
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            if len(x_test) > 0:
                x_test = np.concatenate((x_test,x))
            else:
                x_test=x
    
    # 為vgg下的函式，目的是為了轉成 VGG 的 input 格式
    x_test = preprocess_input(x_test)
    
    # include_top=False，表示會載入 VGG16 的模型，不包括加在最後3層的卷積層，通常是取得 Features (1,7,7,512)
    model = VGG16(weights='imagenet', include_top=False) 
    
    #?
    # 萃取特徵
    features = model.predict(x_test)
    
    # 計算相似矩陣
    features_compress = features.reshape(len(y_test),7*7*512)
    sim = cosine_similarity(features_compress)
    
    print("樣本所在列查詢與其他圖檔的相似度:\n",sim)
    
    """
    透過"python 程式名稱.py 參數"的方式傳入參數
    #inputNo = int(sys.argv[1])   
    #若用此方法，需
    # 1.在最後的最外面加入"if __name__ == '__main__':main()"
    # 2.將本函式改為main()
    """
    top = np.argsort(-sim[inputNo], axis=0)[1:3]

    # 取得最相似的前2名序號
    recommend = [y_test[i] for i in top]
    #?
    
    print("與",y_test[inputNo],"最相似的前2名是:")
    print(recommend)

number=int(input("請輸入要查詢第幾個圖檔(0~n-1):"))
main_func(number)