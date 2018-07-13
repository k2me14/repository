import numpy as np

from keras.models import load_model
import fx_config as fxconf
import fx_preprocess as fxproc
import fx_generator as fxgen
import fx_predict as fxpred
import gc
import sys

config = fxconf.FxConfig()


FX_MODEL_SAVE_FILENAME = config.get_model_save_filename()
FX_MODEL_SAVE = config.get_model_save_flag()
FX_RATE_FILENAME = config.get_rate_filename()
FX_MODEL = config.get_model_name()
FX_SCALE_EACH = config.get_scale_each_sequence_flag()
FX_SCALE_ALL = config.get_scale_all_sequence_flag()
FX_SEQUENCE_LEN = config.get_sequence_len()
FX_PREDICT_LEN = config.get_predict_len()

# 評価期間の設定　***訓練期間と同じような要領で指定　trueにすると詳細情報が確認できる***
eval_start = "20160203"
eval_end = "20160204"
eval_output = True

# 評価期間をコマンドラインから受け取ることもできるようにする
# コードに埋め込まれている値を上書き
args = sys.argv
if len(args) == 3:
    eval_start = args[1]
    eval_end = args[2]

# 学習済みモデルをロード
print("PROGRESS : load trained model")
model = load_model(FX_MODEL_SAVE_FILENAME)

# データの読み込み　＆　事前処理
print("PROGRESS : read rate file")
proc = fxproc.FxPreprocessor(FX_RATE_FILENAME)

ret = proc.open_file()
if not ret:
    print("ERROR : cannot open file")
    exit(-1)

col = config.get_target_pair()
dtlst_txt, rate = proc.read_file(col, start=eval_start, end=eval_end, freq=1)

# データの正規化
if FX_SCALE_ALL and FX_SCALE_EACH:
    # 正規化の方法はどちらか片方のみ。両方True担っていた場合はError
    print("ERROR : standard scaling is duplicated")
    exit(-1)
elif FX_SCALE_ALL:
    print("PROGRSS : scaling")
    proc.standard_scaler(rate)

# 時間データを分割
dtlst = proc.transform_dttxt_to_dtobj(dtlst_txt)

del dtlst_txt
gc.collect()

# データのシャッフル
total_len = len(dtlst) - FX_SEQUENCE_LEN - FX_PREDICT_LEN
idx = [i for i in range(total_len)]


# データのジェネレータ初期化
gen_test = None
print("PROGRES : init generator")
if FX_MODEL == "class":
    gen_test = fxgen.FxGeneratorClassifier(dtlst, rate, idx)
elif FX_MODEL == "regression":
    gen_test = fxgen.FxGeneratorRegressor(dtlst, rate, idx)
else:
    print("ERROR : invalid FX_MODEL = {}".format(FX_MODEL))
    exit(-1)

# データのジェネレータを作成
num_batches_per_epoch_test, data_generator_test = gen_test.batch_iter()

# モデルを作成
eval_test = None
if FX_MODEL == "class":
    eval_test = fxpred.FxPredictClassifier(model)
elif FX_MODEL == "regression":
    eval_test = fxpred.FxPredictRegressor(model)
else:
    print("ERROR : invalid FX_MODEL = {}".format(FX_MODEL))
    exit(-1)


# モデル評価1
print("PROGRESS : evaluate")
score = eval_test.evaluate(data_generator_test, num_batches_per_epoch_test)
print("Score = {}".format(score))

# モデル評価2
print("PROGRESS : predict")
eval_test.predict(data_generator_test, num_batches_per_epoch_test)
ratio = eval_test.get_correct_ratio(data_generator_test, num_batches_per_epoch_test)
print("正答率 = {}".format(ratio))

print("=== output summary ===")
print("start, end, ratio, rate[start], rate[end]")
print("{},{},{},{},{}".format(eval_start, eval_end, ratio, rate[0][0], rate[0][-1]))
eval_test.get_result_summary(data_generator_test, num_batches_per_epoch_test)

if eval_output:
    # 予測結果の出力
    print("PROGRESS : output")
    eval_test.write_answer_predict(data_generator_test, num_batches_per_epoch_test)

    # 予測結果の出力
    eval_test.write_predict_value()
