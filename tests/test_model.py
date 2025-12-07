import ai.train_model as train_model 
from ai.dataset_generator import generate_dataset


def test_dataset_shape():
    df = generate_dataset(n_samples=50)
    assert not df.empty
    assert set(df.columns) == {
        "crop",
        "soil_moisture",
        "temperature",
        "humidity",
        "water_lpm2",
    }


def test_model_training_and_prediction():
    info = train_model.train_and_save()
    pipe = info["pipeline"]
    assert "r2" in info
    sample = {
        "crop": ["tomato"],
        "soil_moisture": [30],
        "temperature": [32],
        "humidity": [50],
    }
    import pandas as pd

    X = pd.DataFrame(sample)
    y_pred = pipe.predict(X)
    assert y_pred.shape == (1,)
