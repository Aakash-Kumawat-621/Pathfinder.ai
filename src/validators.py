"""Runtime and testable validation for PathFinder learning paths."""


class PathValidationError(Exception):
    """Raised when a generated path violates a structural safety guarantee."""


def _prereq_map(df):
    return {
        row["course_id"]: [p.strip() for p in str(row["prerequisites"]).split(";") if p.strip()]
        for _, row in df.iterrows()
    }


def validate_catalog(df):
    """Validate IDs, prerequisite references and cycles before the catalog is trusted."""
    ids = list(df["course_id"])
    if len(ids) != len(set(ids)):
        raise PathValidationError("Catalog contains duplicate course IDs.")
    mapping = _prereq_map(df)
    unknown = [(cid, p) for cid, ps in mapping.items() for p in ps if p not in mapping]
    if unknown:
        raise PathValidationError(f"Unknown prerequisite '{unknown[0][1]}' referenced by '{unknown[0][0]}'.")
    state = {}

    def visit(cid):
        if state.get(cid) == 1:
            raise PathValidationError(f"Prerequisite cycle detected at '{cid}'.")
        if state.get(cid) == 2:
            return
        state[cid] = 1
        for p in mapping[cid]:
            visit(p)
        state[cid] = 2

    for cid in mapping:
        visit(cid)
    return True


def validate_prerequisite_order(ordered_ids, df, completed_courses=None):
    completed = set(completed_courses or [])
    prereq_map = _prereq_map(df)
    valid_ids = set(df["course_id"])
    seen = set()
    for cid in ordered_ids:
        if cid not in valid_ids:
            raise PathValidationError(f"'{cid}' is not a real catalog item.")
        if cid in seen:
            raise PathValidationError(f"'{cid}' appears more than once in the path.")
        for prereq in prereq_map.get(cid, []):
            if prereq in completed:
                continue
            if prereq in ordered_ids and prereq not in seen:
                raise PathValidationError(f"Ordering violation: '{cid}' appears before prerequisite '{prereq}'.")
        seen.add(cid)
    return True


def validate_no_duplicate_milestones_out_of_order(milestones):
    rank = {"Foundations": 0, "Core Skills": 1, "Applied / Advanced": 2, "Capstone": 3}
    seen_max = -1
    warnings = []
    for i, milestone in enumerate(milestones):
        r = rank.get(milestone, 1)
        if r < seen_max - 1:
            warnings.append(f"Step {i}: milestone '{milestone}' appears unusually late/out of order.")
        seen_max = max(seen_max, r)
    return warnings
