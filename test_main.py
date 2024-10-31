from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "ML-приложение для определения тональности текста"}


def test_predict_positive():
    response = client.post("/predict/",
                           json={"text": "Я люблю машинное обучение и тесты!"})
    json_data = response.json()

    assert response.status_code == 200
    assert json_data['label'] == 'POSITIVE'


def test_predict_negative():
    response = client.post("/predict/",
                           json={"text": "Я ненавижу распределенные вычисления и тесты!"})
    json_data = response.json()

    assert response.status_code == 200
    assert json_data['label'] == 'NEGATIVE'
