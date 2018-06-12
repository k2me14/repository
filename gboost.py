import pandas as pd
import numpy as np

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")


# 欠損データ前処理 代理データを中央値や最頻値で置き換える

train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")

# 前処理 カテゴリカルデータの数値化　Sexは「male」「female」、Embarkedは「S」「C」「Q」の3つ。これらを数字に変換

train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2


# test も同様に前処理

test["Age"] = test["Age"].fillna(test["Age"].median())
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
test.Fare[152] = test.Fare.median()


#　家族数を、SibSp = SibSp + Parch として使用
train["SibSp"]=train["SibSp"] + train["Parch"]+1
test["SibSp"]=test["SibSp"] + test["Parch"]+1


# scikit-learnのインポートをします
from sklearn.ensemble import GradientBoostingClassifier

# 「train」の目的変数と説明変数の値を取得
target = train["Survived"].values


#　SibSp = SibSp + Parch として使っている
features = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Embarked"]].values

# モデルは勾配ブースティング
forest = GradientBoostingClassifier(n_estimators=55, random_state=9)
forest = forest.fit(features, target)

# testから使う項目の値を取り出す
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Embarked"]].values


# 予測をしてCSVへ書き出す
my_prediction_forest = forest.predict(test_features)
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution_forest = pd.DataFrame(my_prediction_forest, PassengerId, columns=["Survived"])
my_solution_forest.to_csv("forest3.csv", index_label=["PassengerId"])
