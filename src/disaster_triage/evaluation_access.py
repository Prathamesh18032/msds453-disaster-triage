"""Internal authorization passed from a verified readiness receipt to final readers."""

_KEY = object()


class _FinalAccess:
    def __init__(self, key, *, scope, runs, policies):
        if key is not _KEY:
            raise ValueError("final-test access requires verified campaign readiness")
        self.scope = scope.copy()
        self.runs = {run["run_id"]: run.copy() for run in runs}
        self.policies = policies.copy()

    def check_role(self, scope):
        if any(scope.get(key) != value for key, value in self.scope.items()):
            raise ValueError("final-test scope does not match the verified campaign")

    def check_run(self, run):
        expected = self.runs.get(run.get("run_id"))
        if expected is None or any(run.get(key) != value for key, value in expected.items()):
            raise ValueError("final-test run does not match the verified campaign")

    def check_policy(self, run_id, digest):
        if self.policies[run_id] != digest:
            raise ValueError("final-test policy does not match the verified campaign")


def _authorize_final(**verified):
    return _FinalAccess(_KEY, **verified)


def require_final_access(access, scope):
    if type(access) is not _FinalAccess:
        raise ValueError("final-test predictions disabled until Slice 8 selection/scope lock and readiness")
    access.check_role(scope)
