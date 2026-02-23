import joblib
import pandas as pd
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,field_validator
from feature_extractor import extract_features

#Loading model
try:
  model=joblib.load("models/phishing_ml_model.pkl")
  threshold=joblib.load("models/threshold.pkl")
except FileNotFoundError as e:
  print(f"Model file not found: {e}")
  raise SystemExit(1)

app=FastAPI(
  title="Phishing URL Detection API",
  description="API to detect phishing URLs using a trained machine learning model.",
  version="1.0.0"
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],

)

class URLRequest(BaseModel):
  url:str

  @field_validator('url')
  def url_null_check(cls,v):
    v=v.strip()
    if not v:
      raise ValueError("URL cannot be empty")
    if len(v)>2048:
      raise ValueError("URL length exceeds maximum limit of 2048 characters")
    return v
  
class PredictionResponse(BaseModel):
  url:str
  prediction:str
  confidence:float
  is_phishing:bool
  risk_level:str


@app.get("/health")
def health_check():
  return {"status":"ok",
          "model_loaded":"model is not None"}

@app.post("/predict",response_model=PredictionResponse)
def predict(request: URLRequest):
  url=request.url.strip()

  try:
    features=extract_features(url)
  except Exception as e:
    raise HTTPException(
      status_code=500,
      detail=f"Error extracting features: {str(e)}"
    )
  
  try:
    features_df=pd.DataFrame([features])
  except Exception as e:
    raise HTTPException(
      status_code=500,
      detail=f"Feature conversion error: {str(e)}"
    )

  try:
    prob_phishing=model.predict_proba(features_df)[0][1]
  except Exception as e:
    raise HTTPException(
      status_code=500,
      detail=f"Model prediction error: {str(e)}"
    )
  
  is_phishing=bool(prob_phishing>=threshold)
  confidence=round(prob_phishing*100,2)

  if prob_phishing>=0.75:
    risk_level="High"
  elif prob_phishing>=0.45:
    risk_level="Medium"
  else:
    risk_level="Low"
  
  return PredictionResponse(
    url=url,
    prediction="Phishing URL" if is_phishing else "Valid URL",
    confidence=confidence,
    is_phishing=is_phishing,
    risk_level=risk_level
  )


# if __name__=="__main__":
#   import uvicorn
#   uvicorn.run("api:app",host="0.0.0.0",port=8000,reload=True)

