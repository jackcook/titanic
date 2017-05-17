from flask import Flask, request
import random as rnd

from demo import predict

app = Flask(__name__)

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
    return app.send_static_file("survived.html" if survival[0] else "died.html")

if __name__ == "__main__":
    app.run(debug=True)
