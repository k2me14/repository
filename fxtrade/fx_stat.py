import fx_config as fxconf
import fx_preprocess as fxproc
import fx_generator as fxgen
from collections import Counter
import fx_model_classifier as fxmodclass
import fx_model_regressor as fxmodregr
import gc
import math
import numpy as np
import matplotlib.pyplot as plt

def rate_count(rate, low, high):

    count = 0
    rate_len = len(rate)

    for i in range(0, rate_len):
        if low < rate[i] <= high:
            count += 1

    return count


def moving_average(a, n=3):

    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n



# 設定ファイル読み込み
config = fxconf.FxConfig()
FX_RATE_FILENAME = config.get_rate_filename()
FX_SEQUENCE_LEN = config.get_sequence_len()
FX_PREDICT_LEN = config.get_predict_len()

# データの読み込み　＆　事前処理
proc = fxproc.FxPreprocessor(FX_RATE_FILENAME)
#proc = fxproc.FxPreprocessor("./fx-outdata-marge_uptrend.csv")

ret = proc.open_file()
if not ret:
    print("ERROR : cannot open file")
    exit(-1)

col = config.get_target_pair()
dtlst, rate = proc.read_file(col, start="20160128", end="20160129", freq=1)

rate_np = np.array(rate[0])


# データのシャッフル
total_len = len(dtlst) - FX_SEQUENCE_LEN - FX_PREDICT_LEN
idx = [i for i in range(total_len)]


# ジェネレータの生成
gen = None
MODEL_NAME = config.get_model_name()
if MODEL_NAME == "class":
    gen = fxgen.FxGeneratorClassifier(dtlst, rate, idx)
elif MODEL_NAME == "regression":
    gen = fxgen.FxGeneratorRegressor(dtlst, rate, idx)
else:
    print("ERROR")


print("=== the number of samples in each category ===")
ans = gen.evaluate_ans_func()
count = Counter(ans)
if MODEL_NAME == "class":
    col_len = gen.get_ans_col_len()
    for x in range(col_len):
        index = float(x)
        print("{} = {}".format(x, count[index]))
elif MODEL_NAME == "regression":
    keys = count.keys()
    for x in keys:
        print("{} = {}".format(x, count[x]))

print("=== basic statistic info ===")
rate_len = len(rate[0])
rate_max = np.max(rate_np)
rate_min = np.min(rate_np)
rate_mean = np.mean(rate_np)
rate_median = np.median(rate_np)
rate_var = np.var(rate_np)
rate_std = np.std(rate_np)
print("Count rate  : {}".format(rate_len))
print("Max rate    : {}".format(rate_max))
print("Min rate    : {}".format(rate_min))
print("Mean rate   : {}".format(rate_mean))
print("Median rate : {}".format(rate_median))
print("Var rate    : {}".format(rate_var))
print("Std rate    : {}".format(rate_std))

plt.plot(rate_np)
plt.show()


'''
moving_avg = moving_average(rate_np, 3600)
plt.plot(moving_avg)
plt.ylim([110.0, 125.0])
plt.show()
'''


'''
print("=== histogram ===")
# スタージェスの公式
bin = math.ceil(1 + np.log2(rate_len))
width = (rate_max - rate_min) / bin
bin_num = int(bin)

low = rate_min
hist_x = []
hist_y = []
for _ in range(bin_num):
    high = low + width
    count = rate_count(rate[0], low, high)
    hist_x.append("{:.3f}".format(low))
    hist_y.append(count)
    print("{} ~ {} : {}".format(low, high, count))
    low = high

x = [i for i in range(bin_num)]

ax = plt.subplot()
ax.bar(hist_x, hist_y)
ax.set_xticklabels(hist_x,rotation=90,fontsize=10)
plt.tight_layout()
plt.show()
'''


