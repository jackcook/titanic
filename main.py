import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression

def clean(dataset):
    dataset["Title"] = dataset.Name.str.extract(" ([A-Za-z]+)\.", expand=False) \
        .replace(["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"], "Rare") \
        .replace("Mlle", "Miss") \
        .replace("Ms", "Miss") \
        .replace("Mme", "Mrs") \
        .map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}) \
        .fillna(0)

    dataset["Sex"] = dataset["Sex"].map({"female": 1, "male": 0}).astype(int)

    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset["Sex"] == i) & (dataset["Pclass"] == j + 1)]["Age"].dropna()
            age_guess = int(guess_df.median() / 0.5 + 0.5) * 0.5
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), "Age"] = age_guess

    dataset["Age"] = dataset["Age"].astype(int)

    dataset.loc[dataset["Age"] <= 16, "Age"] = 0
    dataset.loc[(dataset["Age"] > 16) & (dataset["Age"] <= 32), "Age"] = 1
    dataset.loc[(dataset["Age"] > 32) & (dataset["Age"] <= 48), "Age"] = 2
    dataset.loc[(dataset["Age"] > 48) & (dataset["Age"] <= 64), "Age"] = 3
    dataset.loc[dataset["Age"] > 64, "Age"] = 4

    dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1
    dataset["IsAlone"] = 0
    dataset.loc[dataset["FamilySize"] == 1, "IsAlone"] = 1

    freq_port = dataset.Embarked.dropna().mode()[0]
    dataset["Embarked"] = dataset["Embarked"].fillna(freq_port)
    dataset["Embarked"] = dataset["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)

    dataset["Fare"].fillna(dataset["Fare"].dropna().median(), inplace=True)
    dataset.loc[dataset["Fare"] <= 7.91, "Fare"] = 0
    dataset.loc[(dataset["Fare"] > 7.91) & (dataset["Fare"] <= 14.454), "Fare"] = 1
    dataset.loc[(dataset["Fare"] > 14.454) & (dataset["Fare"] <= 31), "Fare"] = 2
    dataset.loc[dataset["Fare"] > 31, "Fare"] = 3
    dataset["Fare"] = dataset["Fare"].astype(int)

    return dataset.drop(["Name", "Parch", "SibSp", "FamilySize"], axis=1)

train_df = clean(pd.read_csv("./input/train.csv").drop(["PassengerId", "Ticket", "Cabin"], axis=1))
test_df = clean(pd.read_csv("./input/test.csv").drop(["Ticket", "Cabin"], axis=1))

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId", axis=1).copy()

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)

submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": Y_pred})
submission.to_csv("./output/submission.csv", index=False)
