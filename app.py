from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application


## Route for the home page
@app.route('/')
def index():
    return render_template('index.html') 


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        # Render empty form
        return render_template('home.html', results=None, form_data={})
    else:
        # Collect form data
        form_data = {
            "gender": request.form.get('gender'),
            "ethnicity": request.form.get('ethnicity'),
            "parental_level_of_education": request.form.get('parental_level_of_education'),
            "lunch": request.form.get('lunch'),
            "test_preparation_course": request.form.get('test_preparation_course'),
            "reading_score": request.form.get('reading_score'),
            "writing_score": request.form.get('writing_score'),
        }

        try:
            # Convert the form data into a custom data object
            data = CustomData(
                gender=form_data["gender"],
                race_ethnicity=form_data["ethnicity"],
                parental_level_of_education=form_data["parental_level_of_education"],
                lunch=form_data["lunch"],
                test_preparation_course=form_data["test_preparation_course"],
                reading_score=float(form_data["reading_score"]),
                writing_score=float(form_data["writing_score"])
            )
            pred_df = data.get_data_as_data_frame()
            print(pred_df)
            print("Before Prediction")

            # Make prediction
            predict_pipeline = PredictPipeline()
            print("Mid Prediction")
            results = predict_pipeline.predict(pred_df)
            print(results)
            print("After Prediction")

            # Render the form with results and the previously entered values
            return render_template('home.html', results=results[0], form_data=form_data)

        except Exception as e:
            # Handle any errors during prediction
            print(f"Error during prediction: {e}")
            return render_template('home.html', results="Error occurred during prediction. Please try again.", form_data=form_data)


if __name__ == "__main__":
    app.run(host="0.0.0.0")
