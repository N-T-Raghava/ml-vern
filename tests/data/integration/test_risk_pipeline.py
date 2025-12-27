def test_risk_pipeline_smoke(risk_module, risk_df):
    # minimal pipeline smoke test
    report = risk_module.run_risk_checks(risk_df, target="target", sensitive=["sex"])
    assert isinstance(report, dict)
