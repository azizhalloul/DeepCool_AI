from src.data.simulate import generate_synthetic_run
from src.data.dataset import TimeSeriesDataset

def test_dataset_shapes():
    df = generate_synthetic_run(length_minutes=200, seed=1)
    ds = TimeSeriesDataset(df, input_len=20, pred_len=5)
    x, y = ds[0]
    assert x.shape == (20, 4)
    assert y.shape == (5,)
