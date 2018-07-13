import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# おまじない
#plt.style.use('ggplot')
#font = {'family' : 'meiryo'}
#matplotlib.rc('font', **font)

#plt.plot([1,2,3,4])
#plt.show()

# 平均 50, 標準偏差 10 の正規乱数を1,000件生成
#x = np.random.normal(50, 10, 1000)
# ヒストグラムを出力
#plt.hist(x)
#plt.show()

# 円グラフを描画
#x = np.array([100, 200, 300, 400, 500])
#plt.pie(x)
#plt.show()


# データ読み込み
#train = pd.read_csv("./train.csv").replace("male",0).replace("female",1)
train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")


# 先頭5行を返す
print(train.head())

# 行列数
print(train.shape)

# 件数、平均、標準偏差、最小、MAX値などの情報を確認できる
#print(train.describe())


# 年齢のみ抽出
train_age = train.iloc[:, 5]
#print(train_df.head())
plt.plot(train_age)
plt.show()


#split_data = []
#for survived in [0,1]:
#    split_data.append(train[train.Survived==survived])

#temp = [i["Pclass"].dropna() for i in split_data]
#plt.hist(temp, histtype="barstacked", bins=3)
#plt.show()