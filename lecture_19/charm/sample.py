from flask import Flask, templating, request
import pickle

with open("vect.pickle", "rb") as f:
    vect = pickle.load(f)

with open("nb.pickle", "rb") as f:
    nb = pickle.load(f)

app = Flask("chacha")

@app.route("/")
def home():
    return templating.render_template("index.html")

@app.route("/submit/", methods=["post"])
def submit():
    message = request.form["message"]
    x_data = vect.transform([message]).todense()
    y_data = nb.predict(x_data)
    print(y_data)
    return str(y_data)
