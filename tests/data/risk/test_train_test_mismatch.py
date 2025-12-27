def test_train_test_mismatch_basic(risk_module, train_test_df):
    train, test = train_test_df
    res = risk_module.train_test_mismatch(train, test, cols=["x"])
    assert "x" in res
