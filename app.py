import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import pickle
app = Flask(__name__)
model = pickle.load(open("static/model/gbr_pred_rb", "rb"))


@app.route('/')
# function home
def home():
    return render_template('index.html')


@app.route('/calculate', methods=['POST', 'GET'])
def calculate():
    if request.method == 'POST':
        int_features = request.get_json()
        simple = int_features[0]['simple']
        average = int_features[0]['average']
        complex_type = int_features[0]['complex']
        bobot_simple = int_features[0]['bobot_simple']
        bobot_average = int_features[0]['bobot_average']
        bobot_complex = int_features[0]['bobot_complex']
        total_calculate = (float(simple) * bobot_simple) + (float(average)
                                                            * bobot_average) + (float(complex_type) * bobot_complex)
        results = {'status': 'success', 'total_calculate': total_calculate}
        return jsonify(results)


if __name__ == "__main__":
    app.run(debug=true)


@app.route("/predict", methods=['POST', 'GET'])
def predict():
    # float_features = [float(x) for x in request.form.values()]
    # print(float_features)
    # feature = [np.array(float_features)]
    # print(feature)
    # prediction = model.predict(feature)
    # return render_template("index.html", prediction_text="{}".format(prediction))

    if request.method == "POST":
        # float_features = [float(x) for x in request.get_json()]
        # feature = np.array(float_features).tolist()
        # prediction = model.predict(feature)
        # prediction = model.predict(
        #     np.array(float_features).tolist()).tolist()
        feature = pd.DataFrame(request.get_json())
        prediction = model.predict(feature)
        results = {'status': 'success', 'result': prediction[0]}
        return jsonify(results)


if __name__ == "__main__":
    app.run(debug=true)
