def test_data_drift_numeric(risk_module, baseline_current_df):
    baseline, current = baseline_current_df
    res = risk_module.data_drift(baseline, current, cols=["x"])
    assert "x" in res
    assert "ks_stat" in res["x"] or "error" in res["x"]
