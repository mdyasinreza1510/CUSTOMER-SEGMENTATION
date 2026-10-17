from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)


#model load
model = joblib.load("models/customer_segmentation_model.pkl")
scaler = joblib.load("models/scaler.pkl")


#making route for frontend
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    income = float(request.form["income"])
    spending = float(request.form["spending"])
    purchases = float(request.form["purchases"])

    user_data = np.array([[age, income, spending, purchases]])

    user_data = scaler.transform(user_data)

    cluster = model.predict(user_data)[0]

    labels = {
        0: "Budget Customer",
        1: "Premium Loyal Customer",
        2: "High Value Customer",
        3: "Occasional Customer"
    }

    prediction = labels[cluster]

    return render_template("index.html", prediction=prediction)




if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)