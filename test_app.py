import json
from fastapi.testclient import TestClient
import app as api
from core import ROOT, Repository, PredictionService

client = TestClient(api.app)
house = {"OverallQual": 7, "GrLivArea": 1710, "FirstFlrSF": 856,
         "TotalBsmtSF": 856, "GarageCars": 2, "YearBuilt": 2003,
         "YrSold": 2008, "FullBath": 2, "TotRmsAbvGrd": 8,
         "Neighborhood": "CollgCr", "KitchenQual": "Gd", "CentralAir": "Y"}


def test_valid():
    response = client.post("/predict", json=house)
    assert response.status_code == 200
    assert response.json()["prediction"] > 0


def test_missing():
    data = dict(house)
    data.pop("GrLivArea")
    assert client.post("/predict", json=data).status_code == 422


def test_negative():
    assert client.post("/predict", json=dict(house, GrLivArea=-1)).status_code == 422


def test_repeat():
    assert client.post("/predict", json=house).json() == client.post("/predict", json=house).json()


def test_unknown():
    assert client.post("/predict", json=dict(house, Neighborhood="NewArea")).status_code == 200


def test_batch():
    assert len(client.post("/predict/batch", json=[house, house]).json()) == 2
    assert client.post("/predict/batch", json=[house] * 101).status_code == 422


def test_unavailable(monkeypatch):
    def missing():
        raise FileNotFoundError
    monkeypatch.setattr(api, "load_service", missing)
    assert client.post("/predict", json=house).status_code == 503


def test_reload():
    saved = json.loads((ROOT / "docs/reload.json").read_text())
    service = PredictionService(Repository())
    assert [service.predict_one(row) for row in saved["inputs"]] == saved["predictions"]
