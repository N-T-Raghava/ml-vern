def test_run_risk_checks_end_to_end(
    risk_module, risk_df, baseline_current_df, train_test_df
):
    baseline, current = baseline_current_df
    train, test = train_test_df
    report = risk_module.run_risk_checks(
        risk_df,
        target="target",
        sensitive=["sex"],
        baseline=baseline,
        train=train,
        test=test,
    )
    assert "class_imbalance" in report
    assert "sensitive_imbalance" in report
    assert "sampling_bias" in report
