def test_target_leakage_detects_copy(risk_module, risk_df):
    df = risk_df.copy()
    # create an obvious leakage column
    df["leak"] = df["target"]
    out = risk_module.target_leakage_detection(df, "target", threshold=0.9)
    assert "leak" in out
    assert "correlation" in out["leak"]
