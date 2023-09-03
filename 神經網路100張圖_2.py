"""
#用第一種儲存方式，用此方式載入模型
import numpy as np  

from keras.models import Sequential
from keras.models import model_from_json
with open(r"D:\0902\神經網路100張圖\1_模型儲存\model.config", "r") as text_file:
    json_string = text_file.read()
    model = Sequential()
    model = model_from_json(json_string)
    model.load_weights(r"D:\0902\神經網路100張圖\1_模型儲存\model.weight", by_name=False)
"""

""""""  
#用第二種儲存方式，用此方式載入模型
from keras.models import load_model


# 刪除既有模型變數
#del model 


# 載入模型
model = load_model(r"D:\0902\神經網路100張圖\1_模型儲存\model.h5")
