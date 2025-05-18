from data.load_data import load_and_preprocess_data

def test_data_shapes():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    assert X_train.shape[1] == 4  # 4 features
    assert len(X_train) > 0

def test_no_nan():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    import numpy as np
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
