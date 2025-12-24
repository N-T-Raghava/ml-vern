import tempfile

from sklearn.linear_model import LogisticRegression

from mlvern.version.commit import commit_run, log_commits


def test_commit_and_log():
    with tempfile.TemporaryDirectory() as tmp:
        model = LogisticRegression()
        cid = commit_run(
            tmp, "test commit", model, metrics={"acc": 0.9}, params={"C": 1.0}
        )

        logs = log_commits(tmp)
        assert len(logs) == 1
        assert logs[0]["id"] == cid
