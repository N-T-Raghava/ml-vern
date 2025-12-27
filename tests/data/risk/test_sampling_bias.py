def test_sampling_bias_baseline_current(risk_module, baseline_current_df):
    baseline, current = baseline_current_df
    res = risk_module.sampling_bias(baseline, current, cols=["x", "cat"])
    assert "x" in res and "cat" in res
    assert "pvalue" in res["x"] or "error" in res["x"]
