from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import joblib
from tempfile import NamedTemporaryFile

model = joblib.load("../weights/model.pkl")
scaler = joblib.load("../weights/scaler.pkl")

app = FastAPI()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        with NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp.seek(0)

            data = pd.read_csv(tmp.name).drop(columns=["search_id", "target"])

        scaled_data = scaler.transform(data)

        # Предсказание 1 класса
        predictions = model.predict_proba(scaled_data)[:, 1]

        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

