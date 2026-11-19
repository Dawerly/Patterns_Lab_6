from functools import lru_cache
from typing import Annotated, Literal
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from core import Repository, PredictionService

app = FastAPI(title="House prices")


class House(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    OverallQual: int = Field(ge=1, le=10)
    GrLivArea: float = Field(gt=0)
    FirstFlrSF: float = Field(ge=0)
    TotalBsmtSF: float = Field(ge=0)
    GarageCars: float = Field(ge=0)
    YearBuilt: int = Field(ge=1800, le=2010)
    YrSold: int = Field(ge=2006, le=2010)
    FullBath: int = Field(ge=0)
    TotRmsAbvGrd: int = Field(ge=0)
    Neighborhood: str
    KitchenQual: Literal["Ex", "Gd", "TA", "Fa", "Po"]
    CentralAir: Literal["Y", "N"]

    @model_validator(mode="after")
    def check_years(self):
        if self.YearBuilt > self.YrSold:
            raise ValueError("Рік побудови пізніший за рік продажу")
        return self


class Prediction(BaseModel):
    prediction: float = Field(allow_inf_nan=False)
    currency: str = "USD"
    model_version: str = "v1"


@lru_cache(maxsize=1)
def load_service():
    return PredictionService(Repository())


def get_service():
    try:
        return load_service()
    except Exception as error:
        raise HTTPException(503, "Модель недоступна") from error


@app.get("/health")
def health():
    try:
        load_service()
        return {"model_loaded": True}
    except Exception:
        return {"model_loaded": False}


@app.get("/model/info")
def info(service=Depends(get_service)):
    return service.metadata


@app.get("/features")
def features(service=Depends(get_service)):
    return service.schema


@app.post("/predict", response_model=Prediction)
def predict(house: House, service=Depends(get_service)):
    try:
        return Prediction(prediction=service.predict_one(house.model_dump()))
    except ValueError as error:
        raise HTTPException(503, "Некоректний прогноз") from error


@app.post("/predict/batch", response_model=list[Prediction])
def batch(houses: Annotated[list[House], Field(min_length=1, max_length=100)], service=Depends(get_service)):
    return [predict(house, service) for house in houses]
