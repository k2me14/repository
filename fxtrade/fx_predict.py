import numpy as np
from abc import ABCMeta, abstractmethod


class FxPredict(metaclass=ABCMeta):

    def __init__(self, model, log=True):
        self.__model = model
        self.log = log


    def evaluate(self, test_gen, test_steps):
        # テスト(評価)
        score = self.__model.evaluate_generator(test_gen, test_steps)
        return score


    def predict(self, test_gen, test_steps):
        # 正答率、準正答率（騰落）集計
        self.preds = self.__model.predict_generator(test_gen, test_steps, verbose=1)


    @abstractmethod
    def get_correct_ratio(self, test_gen, test_steps):
        pass


    @abstractmethod
    def write_answer_predict(self, test_gen, test_steps):
        pass


    def write_predict_value(self):
        if self.log:
            print("PROGRESS : write predict value")

        np.savetxt('preds.csv',self.preds, delimiter=',')


class FxPredictClassifier(FxPredict):

    def __init__(self, model, log=True):
        super().__init__(model, log)


    def get_correct_ratio(self, test_gen, test_steps):
        correct = 0
        count = 0
        step = 0

        for d, a in test_gen:
            for i in range(len(a)):
                pred   = np.argmax(self.preds[count, :])
                target = np.argmax(a[i, :])
                count += 1
                if pred == target:
                    correct += 1
            step += 1
            if step == test_steps:
                break

        return 1.0 * correct / len(self.preds)


    def write_answer_predict(self, test_gen, test_steps):
        if self.log:
            print("PROGRESS : write answer & predict value")

        fp = open("pred_result.txt","w")
        fp.write("answer,predict\n")

        count = 0
        step = 0

        for d, a in test_gen:
            for i in range(len(a)):
                pred   = np.argmax(self.preds[count, :])
                target = np.argmax(a[i, :])
                fp.write("{},{}\n".format(target, pred))
                count += 1

            step += 1
            if step == test_steps:
                break

        fp.close()


class FxPredictRegressor(FxPredict):

    def __init__(self, model, log=True):
        super().__init__(model, log)


    def get_correct_ratio(self, test_gen, test_steps):
        correct = 0
        count = 0
        step = 0

        for d, a in test_gen:
            for i in range(len(a)):

                pred = self.preds[count]
                target = a[i]

                if target <= 0 and pred <=0:
                    correct += 1
                elif target >0 and pred>0:
                    correct += 1

                count += 1

            step += 1
            if step == test_steps:
                break


        return 1.0 * correct / len(self.preds)


    def get_result_summary(self, test_gen, test_steps):
        correct = 0
        count = 0
        step = 0

        result_down = [[0 for _ in range(3)] for _ in range(9) ]
        result_up   = [[0 for _ in range(3)] for _ in range(9) ]

        for d, a in test_gen:
            for i in range(len(a)):

                pred = self.preds[count]
                target = a[i]

                for j in range(9):
                    if pred <= (j * -0.1):
                        result_down[j][0] += 1
                        if target < 0 :
                            result_down[j][1] += 1
                        elif target >= 2:
                            result_down[j][2] += 1

                for j in range(9):
                    if pred > (j * 0.1):
                        result_up[j][0] += 1
                        if target > 0 :
                            result_up[j][1] += 1
                        elif target <= -2:
                            result_up[j][2] += 1

                count += 1

            step += 1
            if step == test_steps:
                break

        for i in range(9):
            print("{},{},{},{}".format(-0.1*i, result_down[i][0], result_down[i][1], result_down[i][2]))

        for i in range(9):
            print("{},{},{},{}".format(0.1*i, result_up[i][0], result_up[i][1], result_up[i][2]))


    def write_answer_predict(self, test_gen, test_steps):
        if self.log:
            print("PROGRESS : write answer & predict value")

        fp = open("pred_result.txt", "w")
        fp.write("answer,predict\n")

        count = 0
        step = 0

        for d, a in test_gen:
            for i in range(len(a)):
                pred = self.preds[count]
                target = a[i]
                fp.write("{},{}\n".format(target, pred))

                count += 1

            step += 1
            if step == test_steps:
                break

        fp.close()