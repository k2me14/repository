import fx_config as fxconf
import fx_preprocess as fxproc
import fx_generator as fxgen
import fx_model_classifier as fxmodclass
import fx_model_regressor as fxmodregr
import fx_plot as fxp
import gc
import math
import socket
import random
import sys

# 設定ファイル読み込み
config = fxconf.FxConfig()
FX_MODEL_SAVE_FILENAME = config.get_model_save_filename()
FX_MODEL_SAVE = config.get_model_save_flag()
FX_RATE_FILENAME = config.get_rate_filename()
FX_MODEL = config.get_model_name()
FX_SCALE_EACH = config.get_scale_each_sequence_flag()
FX_SCALE_ALL = config.get_scale_all_sequence_flag()
FX_SEQUENCE_LEN = config.get_sequence_len()
FX_PREDICT_LEN = config.get_predict_len()

# データファイルをコマンドラインから受け取ることもできるようにする
# コードに埋め込まれている値を上書き
args = sys.argv
if len(args) == 2:
    FX_RATE_FILENAME = args[1]


# データの読み込み　＆　事前処理
proc = fxproc.FxPreprocessor(FX_RATE_FILENAME)
#proc = fxproc.FxPreprocessor("./fx-outdata-marge_01040411.csv")
ret = proc.open_file()
if not ret:
    print("ERROR : cannot open file")
    exit(-1)

# 処理対象の通貨ペアを読み込み
col = config.get_target_pair()

# 為替データの読み込み 　***訓練期間（２０１６年の１年分　指定は１ヶ月分くらいまで）　土日の日付を入れると無限ループ freq で　フェッチ間隔を指定、データは１秒間隔である***
dtlst_txt, rate = proc.read_file(col, start="20160128", end="20160129", freq=1)

# データの正規化
if FX_SCALE_ALL and FX_SCALE_EACH:
    print("ERROR : standard scaling is duplicated")
    exit(-1)
elif FX_SCALE_ALL:
    proc.standard_scaler(rate)

# データのシャッフル　なぜ、６０と５を引き算しているのか？？？

#debug
print(len(dtlst_txt))
print(FX_SEQUENCE_LEN)
print(FX_PREDICT_LEN)


total_len = len(dtlst_txt) - FX_SEQUENCE_LEN - FX_PREDICT_LEN
idx = [i for i in range(total_len)]

#debug
print(total_len)

random.shuffle(idx)

#debug
print(idx)

# 時間データをテキスト形式からdatatime型に変換
dtlst = proc.transform_dttxt_to_dtobj(dtlst_txt)

# データが増大するとメモリ不足になるので明示的にGCを実行
del dtlst_txt
gc.collect()

# データの分割
train_len = math.ceil(total_len * 0.9)
idx_training = idx[0:train_len]
idx_validation = idx[train_len:]

# データが増大するとメモリ不足になるので明示的にGCを実行
del idx
gc.collect()

# データのジェネレータ初期化
gen_training = None
gen_validation = None
if FX_MODEL == "class":
    gen_training = fxgen.FxGeneratorClassifier(dtlst, rate, idx_training)
    gen_validation = fxgen.FxGeneratorClassifier(dtlst, rate, idx_validation)
elif FX_MODEL == "regression":
    gen_training = fxgen.FxGeneratorRegressor(dtlst, rate, idx_training)
    gen_validation = fxgen.FxGeneratorRegressor(dtlst, rate, idx_validation)
else:
    print("ERROR : invalid FX_MODEL = {}".format(FX_MODEL))
    exit(-1)

# データのジェネレータを作成
num_batches_per_epoch_training, data_generator_training = gen_training.batch_iter()
num_batches_per_epoch_validation, data_generator_validation = gen_validation.batch_iter()

# モデルのオブジェクト生成
hidden_num = config.get_hidden_num()
sequence_len = config.get_sequence_len()
data_col_len = gen_training.data_col_len
ans_col_len = gen_training.ans_col_len

lstm_model = None
if FX_MODEL == "class":
    lstm_model = fxmodclass.FxClassifier(sequence_len, hidden_num, data_col_len, ans_col_len)
elif FX_MODEL == "regression":
    lstm_model = fxmodregr.FxRegressor(sequence_len, hidden_num, data_col_len, ans_col_len)
else:
    print("ERROR : invalid FX_MODEL = {}".format(FX_MODEL))
    exit(-1)


# モデル生成
lstm_model.create_model()

#import keras.utils
#from keras.utils import plot_model
#plot_model(lstm_model, to_file="model.png")

# トレーニング
epochs = 1
model, history = lstm_model.train(epochs,
                                  data_generator_training,
                                  num_batches_per_epoch_training,
                                  data_generator_validation,
                                  num_batches_per_epoch_validation)


# 訓練したモデルを保存
if FX_MODEL_SAVE:
    model.save(FX_MODEL_SAVE_FILENAME)


# モデルのサマリーを表示
FX_SHOW_MODEL_SUMMARY = config.get_show_model_summary_flag()
if FX_SHOW_MODEL_SUMMARY:
    model.summary()


# historyをプロット
FX_PLOT_HISTORY = config.get_plot_history_flag()
if FX_PLOT_HISTORY:
    host = socket.gethostname()
    outmethod = 'default'
    if host == 'jrirndgcp01':
        outmethod = 'file'

    if FX_MODEL == "class":
        fxp.plot_history_class(history, outmethod)
    elif FX_MODEL == "regression":
        fxp.plot_history_regression(history, outmethod)
    else:
        print("ERROR : invalid FX_MODEL = {}".format(FX_MODEL))
