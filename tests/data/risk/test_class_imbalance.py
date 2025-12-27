def test_class_imbalance_basic(risk_module, risk_df):
    res = risk_module.class_imbalance(risk_df, "target")
    assert "counts" in res
    assert isinstance(res["imbalance_ratio"], float)
