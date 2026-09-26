import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flasgger import Swagger


MODEL_PATH = os.getenv("MODEL_PATH", "/app/superkart_model.joblib")
MODEL_FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]
FEATURE_REFERENCE_YEAR = int(os.getenv("FEATURE_REFERENCE_YEAR", "2024"))

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Train the model and place "
        "superkart_model.joblib in backend_files/."
    )

model = joblib.load(MODEL_PATH)
app = Flask(__name__)
Swagger(app)


def categorize_product_type(product_type):
    if product_type in ["Fruits and Vegetables", "Dairy", "Meat", "Seafood"]:
        return "Perishables"
    if product_type in [
        "Snack Foods",
        "Baking Goods",
        "Breakfast",
        "Canned",
        "Starchy Foods",
        "Bread",
    ]:
        return "Packaged Foods"
    if product_type in ["Household", "Health and Hygiene"]:
        return "Non-Food Essentials"
    if product_type in ["Soft Drinks", "Hard Drinks"]:
        return "Beverages"
    if product_type == "Frozen Foods":
        return "Frozen Foods"
    return "Others"


def prepare_features(frame):
    frame = frame.copy()

    if "Product_Id" in frame and "Product_Id_char" not in frame:
        frame["Product_Id_char"] = frame["Product_Id"].astype(str).str[:2]
    if "Store_Establishment_Year" in frame and "Store_Age_Years" not in frame:
        frame["Store_Age_Years"] = (
            FEATURE_REFERENCE_YEAR - frame["Store_Establishment_Year"]
        )
    if "Product_Type" in frame and "Product_Type_Category" not in frame:
        frame["Product_Type_Category"] = frame["Product_Type"].apply(
            categorize_product_type
        )

    missing = [column for column in MODEL_FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")

    return frame[MODEL_FEATURES]


@app.get("/")
def health():
    return jsonify({"status": "ok", "model_loaded": True})


@app.post("/v1/predict")
def predict_single():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object containing model features."}), 400

    try:
        features = prepare_features(pd.DataFrame([data]))
        prediction = float(model.predict(features)[0])
        return jsonify({"prediction": prediction})
    except Exception as error:
        return jsonify({"error": str(error)}), 400


@app.post("/v1/predictbatch")
def predict_batch():
    uploaded_file = request.files.get("file")
    if uploaded_file is None:
        return jsonify({"error": "Upload a CSV file using the 'file' field."}), 400

    try:
        batch = pd.read_csv(uploaded_file)
        features = prepare_features(batch)
        predictions = model.predict(features)
        return jsonify({str(index): float(value) for index, value in enumerate(predictions)})
    except Exception as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)