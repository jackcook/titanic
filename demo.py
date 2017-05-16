import pandas as pd
import seaborn as sns
import random as rnd

from sklearn.linear_model import LogisticRegression

def predict(sex, age, siblings, parents):
    train_df = train()

    pclass_rnd = rnd.randint(1, 891)
    pclass = 0

    if pclass_rnd <= 216:
        pclass = 1
    elif pclass_rnd <= 400:
        pclass = 2
    else:
        pclass = 3

    embarked_rnd = rnd.randint(1, 891)
    embarked = 0

    if embarked_rnd <= 168:
        embarked = 1
    elif embarked_rnd <= 245:
        embarked = 2
    else:
        embarked = 0

    fare_band = rnd.randint(0, 3)

    data = {
        "Pclass": pclass,
        "Sex": sex,
        "Age": 0 if age <= 16 else 1,
        "Fare": fare_band,
        "Embarked": embarked,
        "Title": 2 if sex == 1 else 1,
        "IsAlone": siblings == 0
    }

    test_df = pd.DataFrame(data=data, index=[0])

    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test = test_df.copy()

    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    return (logreg.predict(X_test)[0] == 1, pclass, fare_band, embarked)

def train():
    train_df = pd.read_csv("./input/train.csv").drop(["PassengerId", "Ticket", "Cabin"], axis=1)

    train_df["Title"] = train_df.Name.str.extract(" ([A-Za-z]+)\.", expand=False) \
        .replace(["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"], "Rare") \
        .replace("Mlle", "Miss") \
        .replace("Ms", "Miss") \
        .replace("Mme", "Mrs") \
        .map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}) \
        .fillna(0)

    train_df = train_df.drop(["Name"], axis=1)

    train_df["Sex"] = train_df["Sex"].map({"female": 1, "male": 0}).astype(int)

    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = train_df[(train_df["Sex"] == i) & (train_df["Pclass"] == j + 1)]["Age"].dropna()
            age_guess = int(guess_df.median() / 0.5 + 0.5) * 0.5
            train_df.loc[(train_df.Age.isnull()) & (train_df.Sex == i) & (train_df.Pclass == j + 1), "Age"] = age_guess

    train_df["Age"] = train_df["Age"].astype(int)

    train_df.loc[train_df["Age"] <= 16, "Age"] = 0
    train_df.loc[(train_df["Age"] > 16) & (train_df["Age"] <= 32), "Age"] = 1
    train_df.loc[(train_df["Age"] > 32) & (train_df["Age"] <= 48), "Age"] = 2
    train_df.loc[(train_df["Age"] > 48) & (train_df["Age"] <= 64), "Age"] = 3
    train_df.loc[train_df["Age"] > 64, "Age"] = 4

    train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1
    train_df["IsAlone"] = 0
    train_df.loc[train_df["FamilySize"] == 1, "IsAlone"] = 1

    train_df = train_df.drop(["Parch", "SibSp", "FamilySize"], axis=1)

    freq_port = train_df.Embarked.dropna().mode()[0]

    train_df["Embarked"] = train_df["Embarked"].fillna(freq_port)
    train_df["Embarked"] = train_df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)

    train_df.loc[train_df["Fare"] <= 7.91, "Fare"] = 0
    train_df.loc[(train_df["Fare"] > 7.91) & (train_df["Fare"] <= 14.454), "Fare"] = 1
    train_df.loc[(train_df["Fare"] > 14.454) & (train_df["Fare"] <= 31), "Fare"] = 2
    train_df.loc[train_df["Fare"] > 31, "Fare"] = 3
    train_df["Fare"] = train_df["Fare"].astype(int)

    return train_df
