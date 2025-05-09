import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib  # For loading the trained model

# Load the trained model (ensure you provide the correct path)
model = joblib.load("Crop_Recommendation.pkl")

# Load or define feature names (adjust based on your dataset)
feature_names = ["Nitrogen", "Phosphorous", "Potassium", "pH", "Temperature", "Humidity", "Rainfall"]


def generate_shap_explanation(input_features, output_path="static/shap_plot.png"):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(np.array(input_features).reshape(1, -1))

    plt.figure()
    shap.summary_plot(shap_values, input_features, feature_names=feature_names, show=False)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return output_path


def generate_lime_explanation(input_features, output_path="static/lime_plot.png"):
    explainer = lime.lime_tabular.LimeTabularExplainer(training_data=np.array([input_features]),
                                                       feature_names=feature_names,
                                                       mode='classification')
    exp = explainer.explain_instance(np.array(input_features), model.predict_proba)
    exp.save_to_file(output_path)
    return output_path
