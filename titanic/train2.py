import pandas as pd
import numpy as np

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")


#print(train.head())

#test_shape = test.shape
#train_shape = train.shape

#print(test_shape)
#print(train_shape)


#print(test.describe())
#train.describe()

# 参考URL https://www.codexa.net/kaggle-titanic-beginner/

def kesson_table(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum() / len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(
        columns={0: '欠損数', 1: '%'})
    return kesson_table_ren_columns


print(kesson_table(train))
print(kesson_table(test))


# 欠損データ前処理 代理データを中央値や最頻値で置き換える

train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")

print(kesson_table(train))

# 前処理 カテゴリカルデータの数値化　Sexは「male」「female」、Embarkedは「S」「C」「Q」の3つ。これらを数字に変換

train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

print(train.head(10))

# test も同様に前処理

test["Age"] = test["Age"].fillna(test["Age"].median())
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
test.Fare[152] = test.Fare.median()

print(test.head(10))


# scikit-learnのインポートをします
from sklearn import tree

# 「train」の目的変数と説明変数の値を取得
target = train["Survived"].values

# 追加となった項目も含めて予測モデルその2で使う値を取り出す
features_two = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# 決定木の作成とアーギュメントの設定
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=1)
my_tree_two = my_tree_two.fit(features_two, target)

# testから「その2」で使う項目の値を取り出す
test_features_2 = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# 「その2」の決定木を使って予測をしてCSVへ書き出す
my_prediction_tree_two = my_tree_two.predict(test_features_2)
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution_tree_two = pd.DataFrame(my_prediction_tree_two, PassengerId, columns=["Survived"])
my_solution_tree_two.to_csv("my_tree_two.csv", index_label=["PassengerId"])
