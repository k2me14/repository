import numpy as np
import fx_config as fc
from sklearn.preprocessing import StandardScaler
from abc import ABCMeta, abstractmethod
from keras.utils import np_utils
import datetime as dt


class FxGenerator(metaclass=ABCMeta):

    def __init__(self, dtlst, rate, idx):
        self.config = fc.FxConfig()
        self.sequence_len = self.config.get_sequence_len()
        self.batch_size   = self.config.get_batch_size()
        self.predict_len  = self.config.get_predict_len()
        self.dtlst = dtlst
        self.rate  = rate
        self.rate_len = len(rate)
        self.idx = idx
        self.out_cur_flag = self.config.get_output_currency_flag()
        self.out_hour_flag = self.config.get_output_hour_flag()
        self.out_min_flag = self.config.get_output_minute_flag()
        self.out_sec_flag = self.config.get_output_second_flag()
        self.out_week_flag = self.config.get_output_weekday_flag()
        self.scale_each_seq_flag = self.config.get_scale_each_sequence_flag()
        self.target_len = len(idx)
        self.data_col_len = self.get_data_col_len()


    def set_param_debug(self, batch_size, predict_len, sequence_len, idx):
        self.batch_size = batch_size
        self.predict_len = predict_len
        self.sequence_len = sequence_len
        self.idx = idx
        self.target_len = len(idx)


    def batch_iter(self):

        num_batches_per_epoch = int(self.target_len / self.batch_size)

        def data_generator():
            while True:
                for batch_num in range(num_batches_per_epoch):
                    data = self.gen_data(batch_num * self.batch_size)
                    ans = self.gen_ans(batch_num * self.batch_size)
                    yield data, ans

        return num_batches_per_epoch, data_generator()


    def __gen_data_for_one_timestep(self, index):

        data = []
        data.append(self.rate[0][index])
        if self.out_cur_flag:
            for i in range(1,self.rate_len):
                data.append(self.rate[i][index])
        if self.out_week_flag:
            data.append(self.dtlst[index].weekday())
        if self.out_hour_flag:
            data.append(self.dtlst[index].hour)
        if self.out_min_flag:
            data.append(self.dtlst[index].minute)
        if self.out_sec_flag:
            data.append(self.dtlst[index].second)

        return data


    def __standard_scaler(self, one_timestep, index):

        data = np.zeros((self.sequence_len, 1))

        for i in range(self.sequence_len):
            data[i][0] = one_timestep[i][index]

        scaler = StandardScaler()
        data_std = scaler.fit_transform(data)

        for i in range(self.sequence_len):
            one_timestep[i][index] = data_std[i][0]


    def __scale_each_sequence(self, one_timestep):

        self.__standard_scaler(one_timestep, 0)
        if self.out_cur_flag:
            for i in range(1,self.rate_len):
                self.__standard_scaler(one_timestep, i)


    def gen_data(self, start):

        batch_data = []
        for i in range(self.batch_size):
            start_index = self.idx[start + i]
            one_seq = []
            for j in range(self.sequence_len):
                one_timestep = self.__gen_data_for_one_timestep(start_index + j)
                one_seq.append(one_timestep)

            if self.scale_each_seq_flag:
                self.__scale_each_sequence(one_seq)

            batch_data.append(one_seq)

        data = np.array(batch_data)

        return data


    def get_data_col_len(self):

        ret = 1

        if self.out_cur_flag:
            for i in range(1,self.rate_len):
                ret += 1
        if self.out_week_flag:
            ret += 1
        if self.out_hour_flag:
            ret += 1
        if self.out_min_flag:
            ret += 1
        if self.out_sec_flag:
            ret += 1

        return ret


    @abstractmethod
    def gen_ans(self, start_index):
        pass

    @abstractmethod
    def get_ans_col_len(self):
        pass

    @abstractmethod
    def evaluate_ans_func(self):
        pass


class FxGeneratorClassifier(FxGenerator):

    THRESHOLD = [-300.0, -0.0014, 0.0, 0.003, 300.0]
    THRESHOLD_UP = 0.1
    THRESHOLD_DOWN = -0.1

    #__count = 0

    def __init__(self, dtlst, rate, idx):
        super().__init__(dtlst, rate, idx)
        self.method = self.config.get_class_method()
        self.ans_col_len = self.get_ans_col_len()


    def gen_ans(self, start_index):
        batch_data = np.zeros(self.batch_size)

        for i in range(self.batch_size):
            target = 0
            base = self.idx[start_index + i] + self.sequence_len - 1
            #print(base)
            if self.method == "updown":
                target = self.__after_updown(base)
            elif self.method == "keep":
                target = self.__keep_moving(base)
            elif self.method == "minmax":
                target = self.__during_minmax(base)
            else:
                print("ERROR : invalid method {}".format(self.method))

            batch_data[i] = target

        ans = np_utils.to_categorical(batch_data, self.ans_col_len)

        return ans


    def get_ans_col_len(self):
        ret = 0
        if self.method == "updown":
            ret = len(self.THRESHOLD) - 1
        elif self.method == "keep":
            ret = 3
        elif self.method == "minmax":
            ret = 3
        else:
            print("ERROR : invalid method {}".format(self.method))

        return ret


    def __keep_moving(self, start):

        target = 1
        tmp = self.rate[0][start+1 : start + self.predict_len + 1]
        tmpNp = np.array(tmp)
        tmpNp = tmpNp - self.rate[0][start]
        if np.all(tmpNp > 0.):
            # 上がり続ける場合
            target = 0
        elif np.all(tmpNp < 0.):
            # 下がり続ける場合
            target = 2

        return target


    def __after_updown(self, start):

        base = self.rate[0][start]
        val = self.rate[0][start + self.predict_len]
        diff = val - base

        target = 0
        for j in range(len(self.THRESHOLD) - 1):
            if self.THRESHOLD[j] < diff <= self.THRESHOLD[j + 1]:
                target = j

        return target


    def __during_minmax(self, start):

        tmp = self.rate[0][start+1 : start + self.predict_len + 1]
        tmpNp = np.array(tmp)
        tmpNp = tmpNp - self.rate[0][start]

        maxFlag = np.any(tmpNp > self.THRESHOLD_UP)
        minFlag = np.any(tmpNp < self.THRESHOLD_DOWN)

        target = 1
        if maxFlag and (not minFlag):
            target = 0
        elif (not maxFlag) and minFlag:
            target = 2

        return target


    def evaluate_ans_func(self):
        ans = np.zeros(self.target_len)

        for i in range(self.target_len):
            target = 0
            if self.method == "updown":
                target = self.__after_updown(i)
            elif self.method == "keep":
                target = self.__keep_moving(i)
            elif self.method == "minmax":
                target = self.__during_minmax(i)
            else:
                print("ERROR : invalid method {}".format(self.method))

            ans[i] = target

        return ans


class FxGeneratorRegressor(FxGenerator):

    # 頑張ってチューニングする　なんだこれ？？？
    THRESHOLD = [-300.0, -0.01, -0.005, -0.001, 0.0, 0.001, 0.005, 0.01, 300.0]
    RET_WEIGHT = [-5, -2, -1, -0.001, 0.001, 1, 2, 5]

    def __init__(self, dtlst, rate, idx):
        super().__init__(dtlst, rate, idx)
        self.ans_col_len = self.get_ans_col_len()


    def gen_ans(self, start_index):
        batch_data = np.zeros(self.batch_size)

        for i in range(self.batch_size):
            base = self.idx[start_index + i] + self.sequence_len - 1
            batch_data[i] = self.__after_updown(base)

        return batch_data


    def get_ans_col_len(self):
        return 1


    def __after_updown(self, start):

        base = self.rate[0][start]
        val = self.rate[0][start + self.predict_len]
        diff = val - base

        target = 0
        for j in range(len(self.THRESHOLD) - 1):
            if self.THRESHOLD[j] < diff <= self.THRESHOLD[j + 1]:
                target = self.RET_WEIGHT[j]

        return target


    def evaluate_ans_func(self):
        ans = np.zeros(self.target_len)

        for i in range(self.target_len):
            ans[i] = self.__after_updown(i)

        return ans



# テストコード
if __name__ == '__main__':

    dtlst = [dt.datetime(2018, 2, 2),
             dt.datetime(2018, 2, 3),
             dt.datetime(2018, 2, 4),
             dt.datetime(2018, 2, 5),
             dt.datetime(2018, 2, 6),
             dt.datetime(2018, 2, 7),
             dt.datetime(2018, 2, 8),
             dt.datetime(2018, 2, 9),
             ]
    rate = [[1.0,2.0,3.0,4.0,1.0,2.0,3.0,4.0],
            [2.0,4.0,6.0,8.0,1.0,2.0,3.0,4.0]]

    print("--- classifier ---")
    gen = FxGeneratorClassifier(dtlst,rate)
    gen.set_param_debug(2, 2, 1)
    d = gen.gen_data(0)
    print(d)
    a = gen.gen_ans(2)
    print(a)
    print("------------------")

    print("--- regressor ---")
    gen_reg = FxGeneratorRegressor(dtlst,rate)
    gen_reg.set_param_debug(2, 1, 1)
    d = gen_reg.gen_data(0)
    print(d)
    a = gen_reg.gen_ans(2)
    print(a)
    print("------------------")


    gen = FxGeneratorRegressor(dtlst, rate)
    gen.set_param_debug(1,1,3)
    step, gen_iter = gen.batch_iter()

    count = 0
    print(step)
    for d, a in gen_iter:
        print(d, a)
        count += 1
        if count == step:
            break

    print("end")

    count = 0
    for d, a in gen_iter:
        print(d, a)
        count += 1
        if count == step:
            break

    print("end")
