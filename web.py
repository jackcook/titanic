from flask import Flask, request
import pandas as pd
import random as rnd

from sklearn.linear_model import LogisticRegression

from main import clean

app = Flask(__name__)
train_df = clean(pd.read_csv("./input/train.csv").drop(["PassengerId", "Ticket", "Cabin"], axis=1))

def predict(sex, age, siblings, parents):
    global train_df

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
    return logreg.predict(X_test)[0] == 1

@app.route("/")
def root():
    return app.send_static_file("form.html")

@app.route("/submit")
def submit():
    first_name = request.args.get("firstname")
    last_name = request.args.get("lastname")

    gender = int(request.args.get("gender"))
    age = int(request.args.get("age"))
    siblings = int(request.args.get("siblings"))
    parents = 2

    survival = predict(gender, age, siblings, parents)
    return app.send_static_file("survived.html" if survival else "died.html")

if __name__ == "__main__":
    app.run()
