from flask import Flask, request, jsonify
import csv
from prediction import get_prediction

app = Flask(__name__)


@app.route("/predict_alphabet", methods = ["POST"])
def predict_data():
    image = request.files.get('alphabet')
    predict = get_prediction(image)
    return jsonify({
        'prediction' : prediction
    }),200
    


# __ => dunder
if (__name__ == "__main__"):
    app.run(debug = True)





