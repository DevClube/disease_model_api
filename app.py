from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load models and other necessary objects
svm_model = joblib.load('svm_model.joblib')
nb_model = joblib.load('nb_model.joblib')
rf_model = joblib.load('rf_model.joblib')
final_svm_model = joblib.load('final_svm_model.joblib')
final_nb_model = joblib.load('final_nb_model.joblib')
final_rf_model = joblib.load('final_rf_model.joblib')
symptom_index = joblib.load('symptom_index.joblib')
encoder_classes = joblib.load('encoder_classes.joblib')

data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder_classes
}
def predictDisease(symptoms):
    symptoms = symptoms.split(",")

    # creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1

    # reshaping the input data and converting it
    # into suitable format for model predictions
    input_data = np.array(input_data).reshape(1,-1)

    # generating individual outputs
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]

    # making final prediction by taking mode of all predictions
    #final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]

	# making final prediction by taking mode of all predictions
    all_predictions = [rf_prediction, nb_prediction, svm_prediction]
    final_prediction = np.unique(all_predictions)[np.argmax(np.unique(all_predictions, return_counts=True)[1])]

    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction":final_prediction
    }
    return predictions
@app.route('/predict', methods=['GET'])
def predict_disease():
    try:
        symptoms = request.args.get('data')
        predictions = predictDisease(symptoms)
        return jsonify(predictions)
    except Exception as e:
        # Log the exception for debugging
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500  # Return a 500 Internal Server Error status

if __name__ == '__main__':
    app.run(debug=True)
# @app.route('/predict', methods=['GET'])
# def predict_disease():
#     try:
#         # Get input data from the query parameter
#         input_data = request.args.get('data')
#
#         if not input_data:
#             return jsonify({'error': 'Input data is missing'})
#
#         # Convert the input data to a list
#         symptoms = input_data.split(',')
#
#         # Create a function to make predictions
#         def make_prediction(model):
#             input_symptoms = np.array(symptoms).reshape(1, -1)
#             return data_dict["predictions_classes"][model.predict(input_symptoms)[0]]
#
#         # Make predictions using the loaded models
#         rf_prediction = make_prediction(final_rf_model)
#         nb_prediction = make_prediction(final_nb_model)
#         svm_prediction = make_prediction(final_svm_model)
#
#         # Take mode of all predictions
#         all_predictions = [rf_prediction, nb_prediction, svm_prediction]
#         final_prediction = np.unique(all_predictions)[np.argmax(np.unique(all_predictions, return_counts=True)[1])]
#
#         predictions = {
#             "rf_model_prediction": rf_prediction,
#             "naive_bayes_prediction": nb_prediction,
#             "svm_model_prediction": svm_prediction,
#             "final_prediction": final_prediction
#         }
#
#         return jsonify(predictions)
#     except Exception as e:
#         return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
