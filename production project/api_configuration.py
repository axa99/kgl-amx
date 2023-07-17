import pandas as pd
from pycaret.classification import ClassificationExperiment
from fastapi import FastAPI
import uvicorn

# Load trained Pipeline
s = ClassificationExperiment()
model = s.load_model("./models_saved/tuned_cat_boost_07_11_2023")
endapi_df = pd.read_csv("./data/endapi_df.csv", index_col=0)
endapi_df = endapi_df.T


# Define predict function
async def predict(
    D_64,
):
    # concatenate with endapi_df
    endapi_df["D_64"] = D_64
    data = endapi_df
    predictions = s.predict_model(
        model,
        data=data,
    )
    return {"prediction": int(predictions["target"][0])}


# Create the app object
app = FastAPI()


# Define root endpoint
@app.get("/")
async def root_endpoint():
    return {"message": "Welcome to the API!"}


# Define predict endpoint
@app.post("/predict")
async def predict_endpoint(
    D_64,
):
    return await predict(
        D_64,
    )


def start_server():
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    start_server()
