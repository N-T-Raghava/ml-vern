def test_sensitive_attribute_imbalance_basic(risk_module, risk_df):
    res = risk_module.sensitive_attribute_imbalance(risk_df, ["sex"])
    assert "sex" in res
    assert isinstance(res["sex"], dict)
