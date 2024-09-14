import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

loaded_rf_classifier = joblib.load('random_forest_model.pkl')
loaded_label_encoder = joblib.load('label_encoder.pkl')

feature_data = [[9.17,3.06,3.17]]

y_pred = loaded_rf_classifier.predict(feature_data)

# Decode the prediction using LabelEncoder
prediction = loaded_label_encoder.inverse_transform(y_pred)[0]

print(prediction)