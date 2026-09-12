from fastapi import FastAPI, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
from fastapi.staticfiles import StaticFiles

app = FastAPI() # FastAPI application object
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
model = joblib.load("model.pkl")

#Homepage
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    result = request.query_params.get("result")
    probability = request.query_params.get("prob")
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
        "result": result,
        "probability": probability
    })

#Prediction route (from form)
@app.post("/predict")
def predict(
    request: Request,
    radius_mean: float = Form(...),  # (...) means the field is required and cannot be skipped.
    texture_mean: float = Form(...),
    perimeter_mean: float = Form(...),
    area_mean: float = Form(...),
    smoothness_mean: float = Form(...),
    compactness_mean: float = Form(...),
    concavity_mean: float = Form(...),
    concave_points_mean: float = Form(...), 
    symmetry_mean: float = Form(...),
    fractal_dimension_mean: float = Form(...),
    radius_se: float = Form(...), 
    texture_se: float = Form(...), 
    perimeter_se: float = Form(...), 
    area_se: float = Form(...), 
    smoothness_se: float = Form(...),
    compactness_se: float = Form(...), 
    concavity_se: float = Form(...), 
    concave_points_se: float = Form(...),
    symmetry_se: float = Form(...),
    fractal_dimension_se: float = Form(...),
    radius_worst: float = Form(...), 
    texture_worst: float = Form(...),   
    perimeter_worst: float = Form(...), 
    area_worst: float = Form(...), 
    smoothness_worst: float = Form(...),
    compactness_worst: float = Form(...),
    concavity_worst: float = Form(...), 
    concave_points_worst: float = Form(...), 
    symmetry_worst: float = Form(...),
    fractal_dimension_worst: float = Form(...)
):
    input_data = pd.DataFrame([{
        "radius_mean": radius_mean,
        "texture_mean": texture_mean,
        "perimeter_mean": perimeter_mean,
        "area_mean": area_mean, 
        "smoothness_mean": smoothness_mean, 
        "compactness_mean": compactness_mean, 
        "concavity_mean": concavity_mean,
        "concave_points_mean": concave_points_mean, 
        "symmetry_mean": symmetry_mean, 
        "fractal_dimension_mean": fractal_dimension_mean,
        "radius_se": radius_se, 
        "texture_se": texture_se, 
        "perimeter_se": perimeter_se, 
        "area_se": area_se, 
        "smoothness_se": smoothness_se,
        "compactness_se": compactness_se, 
        "concavity_se": concavity_se, 
        "concave_points_se": concave_points_se, 
        "symmetry_se": symmetry_se,
        "fractal_dimension_se": fractal_dimension_se,
        "radius_worst": radius_worst, 
        "texture_worst": texture_worst,        
        "perimeter_worst": perimeter_worst, 
        "area_worst": area_worst, 
        "smoothness_worst": smoothness_worst,
        "compactness_worst": compactness_worst, 
        "concavity_worst": concavity_worst, 
        "concave_points_worst": concave_points_worst, 
        "symmetry_worst": symmetry_worst, 
        "fractal_dimension_worst": fractal_dimension_worst
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    result = "Malignant (cancer)" if prediction == 1 else "Benign (No Cancer)"

    return RedirectResponse(
        url=f"/?result={result}&prob={probability}",
        status_code=303
    )
