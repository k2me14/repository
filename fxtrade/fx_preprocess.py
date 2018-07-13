import datetime
from sklearn.preprocessing import StandardScaler
import numpy as np
import gc
import math

class FxPreprocessor:

    def __init__(self, filename):
        self.__input_filename = filename


    def open_file(self):
        flag = True

        try:
            self.__fin = open(self.__input_filename, "r")
        except IOError as e:
            print("File \"{0}\" is not found.".format(e))
            flag = False

        return flag


    def __is_equal_date(self, dateFromFile, dateFromArg):
        return dateFromFile.startswith(dateFromArg)


    def __is_specified_second(self, dateFromFile, secondFromArg):
        slice = dateFromFile[-2:]
        return slice == secondFromArg



    def __search_start_line(self, start):
        line = " "
        while line:
            line = self.__fin.readline()
            items = line[:-1].split(',')
            if self.__is_equal_date(items[0], start):
                break

        return line


    def read_file(self, col, start=' ', end=' ', freq=1):
        '''rateファイルを読み込む

        Keyword arguments:
            col -- レートを読み込む列番号を指定
            start -- 読み込みを開始する日付を指定 (形式:YYYYmmdd)
            end -- 読み込みを終了する日付を指定 (形式:YYYYmmdd)
            freq -- 読み込みの間隔を指定 (例: 1=全ての行、2=隔行)
        '''

        # 変数初期化
        rate_len = len(col)                     # 読み取るレート数
        dtlst = []                              # 読み取った日付データのリスト
        rate = [[] for _ in range(rate_len)]    # 読み取ったレート情報のリスト
        count = 0                               # 読み取り間隔を制御するための変数
        read_len = 0                            # 読み取った行数を表示するための変数

        # startが見つかるまでファイル読み込み
        if start != ' ':
            line = self.__search_start_line(start)
            if line == "":
                print("ERROR : start date ({}) not found".format(start))
        else:
            line = self.__fin.readline()

        # ファイルの読み込み
        while line:
            if count % freq == 0:
                items = line[:-1].split(',')  # 改行文字削除

                # endに達したかを判定
                if self.__is_equal_date(items[0], end):
                    break

                dtlst.append(items[0])
                for i, x in enumerate(col):
                    rate[i].append(float(items[x]))

                read_len += 1

                if read_len % 100000 == 0:
                    print("PROGRESS : line read = {}".format(read_len))

            count += 1
            line = self.__fin.readline()

        # 結果の表示
        print("INFO : the number of line read from {} = {}".format(self.__input_filename, read_len))

        return dtlst, rate


    def read_file_specified_second(self, col, second, start=' ', end=' '):
        '''rateファイルを読み込む。その時に引数で指定された秒ポイントのデータのみ読み込む

        Keyword arguments:
            col -- レートを読み込む列番号を指定
            second -- 指定された秒 (形式:dd)
            start -- 読み込みを開始する日付を指定 (形式:YYYYmmdd)
            end -- 読み込みを終了する日付を指定 (形式:YYYYmmdd)
        '''

        # 変数初期化
        rate_len = len(col)                     # 読み取るレート数
        dtlst = []                              # 読み取った日付データのリスト
        rate = [[] for _ in range(rate_len)]    # 読み取ったレート情報のリスト
        read_len = 0                            # 読み取った行数を表示するための変数

        # startが見つかるまでファイル読み込み
        if start != ' ':
            line = self.__search_start_line(start)
            if line == "":
                print("ERROR : start date ({}) not found".format(start))
        else:
            line = self.__fin.readline()

        # ファイルの読み込み
        while line:
            items = line[:-1].split(',')  # 改行文字削除

            # endに達したかを判定
            if self.__is_equal_date(items[0], end):
                break

            # 指定された秒かを判定
            if self.__is_specified_second(items[0], second):
                dtlst.append(items[0])
                for i, x in enumerate(col):
                    rate[i].append(float(items[x]))

                read_len += 1

                if read_len % 100000 == 0:
                    print("PROGRESS : line read = {}".format(read_len))

            line = self.__fin.readline()

        # 結果の表示
        print("INFO : the number of line read from {} = {}".format(self.__input_filename, read_len))

        return dtlst, rate


    def transform_dttxt_to_dtobj(self, dttxt):
        dtobj = []
        for x in dttxt:
            dt = datetime.datetime.strptime(x, '%Y%m%d %H:%M:%S')
            dtobj.append(dt)

        return dtobj


    def standard_scaler(self, rates):

        import fx2.fx_config as fxconf

        config = fxconf.FxConfig()
        SAVE_FLAG = config.get_scale_save_flag()
        SAVE_FILENAME = config.get_scale_save_filename()

        if SAVE_FLAG:
            fin = open(SAVE_FILENAME, "w")

        for i in range(len(rates)):
            ratesNp = np.array(rates[i])
            reshaped = ratesNp.reshape(-1, 1)  # StandardScallerの入力形式に合わせる
            scaler = StandardScaler()
            scaler.fit(reshaped)

            if SAVE_FLAG:
                self.__save_scaler_value(fin, scaler,i)

            dataStd = scaler.transform(reshaped)

            for j in range(0,len(rates[i])):
                rates[i][j] = dataStd[j][0]

        del ratesNp
        del reshaped
        gc.collect()

        if SAVE_FLAG:
            fin.close()



    def manual_standard_scaler(self, rates):

        import fx2.fx_config as fxconf

        config = fxconf.FxConfig()
        SAVE_FLAG = config.get_scale_save_flag()
        SAVE_FILENAME = config.get_scale_save_filename()

        if not SAVE_FLAG:
            print("ERROR : please make sure the statistic values are saved in the file")
            return

        fin = open(SAVE_FILENAME, "r")

        for i in range(len(rates)):
            mean , std = self.__read_scaler_value(fin)
            for j in range(len(rate[i])):
                rate[i][j] = (rate[i][j] - mean) / std

        fin.close()


    def __save_scaler_value(self, fin, scaler, col):
        fin.write("{},{},{},{},{}\n".format(col, scaler.mean_[0], scaler.var_[0], scaler.scale_[0], scaler.n_samples_seen_))


    def __read_scaler_value(self, fin):
        line = fin.readline()
        items = line[:-1].split(',')  # 改行文字削除

        mean = float(items[1])
        var = float(items[2])
        std = math.sqrt(var)

        return mean, std


    def shuffle(self, dtlst, rates):

        order = np.arange(len(dtlst))
        np.random.shuffle(order)

        dtlst_s = []
        rates_s = [[] for _ in range(len(rates))]

        for i in range(len(dtlst)):
            dtlst_s.append(dtlst[order[i]])
            for j in range(len(rates)):
                rates_s[j].append(rates[j][order[i]])

        del order
        gc.collect()

        return dtlst_s, rates_s


    def close_file(self):
        self.__fin.close()

        return True


# テストコード
if __name__ == '__main__':

    import fx2.fx_config as fxconf

    config = fxconf.FxConfig()
    FX_RATE_FILENAME = config.get_rate_filename_for_debug()

    process = FxPreprocessor(FX_RATE_FILENAME)

    ret = process.open_file()
    if not ret:
        print("ERROR : cannot open file")
        exit(-1)

    col = config.get_target_pair()
    dtlst, rate = process.read_file(col,start="20170103", end="20170106", freq=2)

    print("=== date time list ===")
    print(dtlst)

    print("=== rate list ===")
    print(rate)

    process.close_file()
    del process



    process = FxPreprocessor(FX_RATE_FILENAME)

    ret = process.open_file()
    if not ret:
        print("ERROR : cannot open file")
        exit(-1)

    col = config.get_target_pair()
    dtlst, rate = process.read_file(col, start="20170106", end="20170107")

    print("=== date time list ===")
    print(dtlst)

    print("=== rate list ===")
    print(rate)

    process.close_file()

    print("=== scale ===")
    process.standard_scaler(rate)
    print(rate)

    rateNp = np.array(rate[0])
    mean = np.mean(rateNp)
    var = np.var(rateNp)

    print("mean = {}, var = {}".format(mean, var))

    ret = process.open_file()
    dtlst, rate = process.read_file(col, start="20170106", end="20170107")
    process.close_file()

    print("=== manual scale ===")
    process.manual_standard_scaler(rate)
    print(rate)

    rateNp = np.array(rate[0])
    mean = np.mean(rateNp)
    var = np.var(rateNp)

    print("mean = {}, var = {}".format(mean, var))
