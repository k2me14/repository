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
dtlst_txt, rate = proc.read_file(col, start="20161114", end="20161205", freq=1)


# ファイルオープン 追記 改行しないバグあり、、、
f = open('fx-outdata-marge_uptrend.csv', 'a')
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