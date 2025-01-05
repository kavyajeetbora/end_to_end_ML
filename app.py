import requests
import pickle
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        ## This is where we take the user inputs and parse them to CustomData Format: dataframe
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethinicity=request.form.get("ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=int(request.form.get("reading_score")),
            writing_score=int(request.form.get("writing_score")),
        )

        ## Convert to dataframe
        input_df = data.get_data_as_frame()

        ## Create a PredictPipeline
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(features=input_df)

        return render_template("home.html", results=prediction[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0")
