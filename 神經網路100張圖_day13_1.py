from nltk.tokenize import word_tokenize
# 測試字句
sent = "the the the dog, dog some other words that we do not care about"
# 取出每個單字

lists=[]
for word in word_tokenize(sent):
    lists.append(word)

#得到結果為 ['the', 'the', 'the', 'dog', ',', 'dog', 'some', 'other', 'words', 'that', 'we', 'do', 'not', 'care', 'about']
# 去除重複，並排序
vacabulary = sorted(set(lists)) 
print("vacabulary:")
print(vacabulary)
#得到結果為 [',', 'about', 'care', 'do', 'dog', 'not', 'other', 'some', 'that', 'the', 'we', 'words']
# 求得每個單字的出現頻率
import nltk
freq = nltk.FreqDist(lists)
#得到結果為 FreqDist({'the': 3, 'dog': 2, 'care': 1, 'some': 1, 'other': 1, ',': 1, 'we': 1, 'that': 1, 'words': 1, 'about': 1, ...})

# 作圖
freq.plot()

stopwords=[",","the"]
# 去除 Stop Words
lists=[]
for word in word_tokenize(sent):
    if word not in stopwords:
        lists.append(word)

#得到結果為 ['dog', 'dog', 'some', 'other', 'words', 'that', 'we', 'do', 'not', 'care', 'about']
# 記得載入 WordNet 語料庫
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
# 要指定單字詞性(pos)
print(wnl.lemmatize('ate', pos='v')) # 得到 eat
print(wnl.lemmatize('better', pos='a')) # 得到 good
print(wnl.lemmatize('dogs')) # 得到 dog
# 若要自動取得單字詞性(pos)，請參考 http://www.zmonster.me/2016/01/21/lemmatization-survey.html。