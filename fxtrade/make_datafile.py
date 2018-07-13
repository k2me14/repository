import fx_config as fxconf
import fx_preprocess as fxproc
import csv
import numpy as np
import matplotlib.pyplot as plt

config = fxconf.FxConfig()
FX_RATE_FILENAME = config.get_rate_filename()

# データの読み込み　＆　事前処理
proc = fxproc.FxPreprocessor(FX_RATE_FILENAME)
ret = proc.open_file()
if not ret:
    print("ERROR : cannot open file")
    exit(-1)

# 処理対象の通貨ペアを読み込み
col = config.get_target_pair()

# 為替データの読み込み 　***訓練期間（２０１６年の１年分　指定は１ヶ月分くらいまで）　土日の日付を入れると無限ループ freq で　フェッチ間隔を指定、データは１秒間隔である***
dtlst_txt, rate = proc.read_file(col, start="20160128", end="20160129", freq=1)


plt.plot(rate[0])
plt.show()

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


rate_diff = []
for i in range(1,len(dtlst_txt)):
    rate_diff.append(rate[0][i]-rate[0][i-1])

print(max(rate_diff))
print(min(rate_diff))

data = np.array(rate_diff)
# 平均
print(np.mean(data))
# 標準偏差
print(np.std(data))


# ファイルオープン
f = open('fx-outdata-marge_temp.csv', 'w')
writer = csv.writer(f, lineterminator='\n')

for i in range(len(dtlst_txt)):
    csvlist = []
    csvlist.append(dtlst_txt[i])
    csvlist.append(rate[0][i])
    # ユーロのカラムをダミー入力
    csvlist.append(1)
    writer.writerow(csvlist)

# ファイルクローズ
f.close()