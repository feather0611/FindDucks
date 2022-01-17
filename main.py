import math
import operator
import cv2 as cv
import numpy as np
import pandas as pd

# 計算高斯機率密度函數
def probability(val, mean, stdev):  
    val = float(val)
    gaus = math.exp(-(math.pow(val - mean, 2.0) / (2.0 * math.pow(stdev, 2))))  
    return (1.0 / (math.sqrt(2.0 * math.pi) * stdev)) * gaus

# 利用計算得到的高斯機率密度函數在貝氏分類器中計算機率，並回傳最有可能的結果
def predict(inputVector, distribution): 
    probabilities = {}
    for cla, dist in distribution.items():
        probabilities[cla] = 1
        for i in range(len(dist)):
            mean, std = dist[i]
            x = inputVector[i]
            probabilities[cla] *= probability(x, mean, std)
            
    # 取得機率最高的類別
    max_key = max(probabilities.items(), key=operator.itemgetter(1))[0]

    return max_key

# 讀取建立模型用的圖片
train_img = cv.imread('01duck.jpg')
train_h, train_w = train_img.shape[:2]
train_data = []

# 為每個pixel加上label，若r, g, b三個值皆>220，則標記為鴨子pixel
for y in range(train_h):
    for x in range(train_w):
        b, g, r = train_img[y,x]
        if (b>=220 and g>=220) and r>=220:
            cla = 1
        else:
            cla = 0
        result = {'r':r, 'g':g, 'b':b, 'class':cla}
        train_data.append(result)
        
# 生成一個dataframe
df = pd.DataFrame(train_data)
print('We have ', len(df[df['class'] == 1]), ' pixels are ducks.')
print('We have ', len(df[df['class'] == 0]), ' pixels are background.')

# 重新改變資料結構
generated_by_class = {0:[],1:[]}

for i in range(2):
    rows = df[df['class']==i]
    for j in range(len(rows)):
        x=list(rows.iloc[j])
        generated_by_class[i].append(x)
        
# 計算出兩個類別中所有R, G, B值的平均以及標準差
dist = {}
for cla, dat in generated_by_class.items():
    summary = [(np.mean(attr), np.std(attr)) for attr in
                 zip(*dat[1:-2])]
    del summary[-1]
    dist[cla] = summary
    
print(dist)
    
# 讀取要預測的圖片
test_img = cv.imread('full_duck.jpg')
test_h, test_w = test_img.shape[:2]
# 將圖片轉換為二維的陣列，並另外將pixel的位置儲存下來
test_data = []
test_pos = []

for y in range(test_h):
    for x in range(test_w):
        b, g, r = test_img[y,x]
        test_pos.append([x, y])
        test_data.append([float(r), float(g), float(b)])

# 以迴圈方式將data送入預測
predictions = []
for i in range(len(test_data)):
    result = predict(test_data[i], dist)
    predictions.append(result)
    
# 根據預測所得到的結果，生成一張圖片
output_img = np.zeros([test_h,test_w,3])

for i in range(len(test_pos)):
    x, y = test_pos[i]
    if predictions[i]==1:
        output_img[y,x,0]=255
        output_img[y,x,1]=255
        output_img[y,x,2]=255
    else:
        output_img[y,x,0]=0
        output_img[y,x,1]=0
        output_img[y,x,2]=0
        
cv.imwrite('result.jpg', output_img)