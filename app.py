from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import json
import os
import shutil

from src.pipline.prediction_pipeline import APSSensorDataFrame
from src.pipline.prediction_pipeline import APSSensorPredictor

app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="web_app/static"), name="static")

# Configure templates
templates = Jinja2Templates(directory="web_app/templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary file
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load the JSON data from the file
        with open(file_path, "r") as json_file:
            dictionary = json.load(json_file)

        # Clean up the temporary file
        os.remove(file_path)

        # Create a DataFrame from the dictionary
        obj = APSSensorDataFrame(dictionary=dictionary)
        input_df = obj.final_input_data()

        # Make a prediction
        obj2 = APSSensorPredictor()
        prediction_result = obj2.predict(input_df)
        prediction = "No" if prediction_result == 0 else "Yes"

        return templates.TemplateResponse("result.html", {"request": request, "prediction": prediction})

    except Exception as e:
        return templates.TemplateResponse("result.html", {"request": request, "error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)