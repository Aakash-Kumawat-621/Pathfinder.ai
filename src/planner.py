"""Planning primitives for PathFinder.

The curriculum planner is graph-first: prerequisite dependencies are solved before
personalization scores are allowed to choose among currently-ready courses.
"""
from collections import defaultdict


class PrerequisiteDAGSolver:
    """Deterministic prerequisite DAG with validation and topological traversal."""
    def __init__(self, catalog):
        self.nodes = set()
        self.prereq = {}
        for _, row in catalog.iterrows():
            cid = str(row["course_id"])
            parents = [p.strip() for p in str(row.get("prerequisites", "")) .split(";") if p.strip()]
            self.prereq[cid] = parents
            self.nodes.add(cid)
        unknown = sorted({p for ps in self.prereq.values() for p in ps if p not in self.nodes})
        if unknown:
            raise ValueError("Unknown prerequisite(s): " + ", ".join(unknown))
        self.children = defaultdict(list)
        for child, parents in self.prereq.items():
            for parent in parents:
                self.children[parent].append(child)
        for children in self.children.values():
            children.sort()
        self._assert_acyclic()

    def _assert_acyclic(self):
        indegree = {n: 0 for n in self.nodes}
        for child, parents in self.prereq.items():
            indegree[child] = len(parents)
        q = sorted(n for n, d in indegree.items() if d == 0)
        seen = 0
        while q:
            n = q.pop(0); seen += 1
            for child in self.children.get(n, []):
                indegree[child] -= 1
                if indegree[child] == 0:
                    q.append(child); q.sort()
        if seen != len(self.nodes):
            raise ValueError("Prerequisite cycle detected")

    def ancestors(self, targets):
        out = set(); stack = list(targets)
        while stack:
            n = stack.pop()
            for p in self.prereq.get(n, []):
                if p not in out:
                    out.add(p); stack.append(p)
        return out

    def topological_order(self, targets=None, satisfied=None, priority=None):
        """Return a deterministic, cycle-free legal sequence.

        ``priority`` only breaks ties among nodes whose prerequisites are already
        satisfied; it can never violate dependency order.
        """
        satisfied = set(satisfied or [])
        targets = list(targets) if targets is not None else sorted(self.nodes)
        required = set(targets) | self.ancestors(targets)
        required -= satisfied
        priority = priority or {}
        indegree = {n: sum(p in required for p in self.prereq.get(n, [])) for n in required}
        key = lambda n: (-float(priority.get(n, 0)), n)
        ready = sorted([n for n, d in indegree.items() if d == 0], key=key)
        result = []
        while ready:
            n = ready.pop(0); result.append(n)
            for child in self.children.get(n, []):
                if child in indegree:
                    indegree[child] -= 1
                    if indegree[child] == 0:
                        ready.append(child); ready.sort(key=key)
        if len(result) != len(required):
            raise ValueError("DAG traversal failed: unresolved prerequisite dependency")
        return result



class CostWeightedAStar:
    """Cost-aware A* over the prerequisite DAG.

    The graph remains dependency-safe; A* chooses among legal frontier nodes using
    estimated study cost plus prerequisite friction. Baseline proficiency lowers
    the effective cost of courses covering already-demonstrated skills.
    """
    def __init__(self, catalog, dag_solver=None):
        self.catalog = catalog
        self.dag = dag_solver
        self.rows = {str(r["course_id"]): r for _, r in catalog.iterrows()}
        self.prereq = {cid: [p.strip() for p in str(r.get("prerequisites", "")).split(";") if p.strip()]
                       for cid, r in self.rows.items()}
        self.children = defaultdict(list)
        for child, parents in self.prereq.items():
            for parent in parents:
                self.children[parent].append(child)
        for v in self.children.values(): v.sort()

    def _cost(self, cid, baseline):
        row = self.rows[cid]
        hours = max(0.25, float(row.get("duration_hours", 1)))
        skill = str(row.get("skill_tag", "")).strip().lower()
        prof = float(baseline.get(skill, 0))
        # proficiency 0..3: higher evidence reduces expected study effort.
        mastery_discount = min(0.55, max(0.0, prof / 3.0) * 0.55)
        friction = 0.0
        for parent in self.prereq.get(cid, []):
            pskill = str(self.rows.get(parent, {}).get("skill_tag", "")).lower()
            pprof = float(baseline.get(pskill, 0))
            friction += 0.75 if pprof < 1 else 0.25 if pprof < 2 else 0.0
        return hours * (1.0 - mastery_discount) + friction

    def _heuristic(self, remaining, baseline):
        # Admissible lower bound: sum of minimum effective course costs. It never
        # overestimates because prerequisite friction is omitted here.
        if not remaining:
            return 0.0
        return sum(max(0.0, float(self.rows[c].get("duration_hours", 1))) *
                   (1.0 - min(0.55, max(0.0, float(baseline.get(str(self.rows[c].get("skill_tag", "")).lower(), 0)) / 3.0) * 0.55))
                   for c in remaining)

    def search(self, targets, satisfied=None, baseline=None, priority=None):
        """Cost-weighted A* frontier traversal with deterministic dependency safety.

        Every expansion is restricted to currently-ready DAG nodes. The score is
        f(n)=g(n)+h(n), where g is observed study cost/friction and h is the
        remaining lower-bound workload. This gives an A*-style cost-aware sequence
        without enumerating factorially many equivalent topological orders.
        """
        import heapq
        baseline = {str(k).lower(): float(v) for k, v in (baseline or {}).items()}
        satisfied = set(satisfied or [])
        priority = priority or {}
        required = set(targets)
        stack = list(targets)
        while stack:
            n = stack.pop()
            for parent in self.prereq.get(n, []):
                if parent not in required:
                    required.add(parent); stack.append(parent)
        required -= satisfied
        done = set()
        result = []
        g = 0.0
        while len(done) < len(required):
            ready = [cid for cid in required - done
                     if all(parent in done or parent in satisfied for parent in self.prereq.get(cid, []))]
            if not ready:
                raise ValueError("No legal prerequisite-safe path exists")
            heap = []
            remaining = required - done
            for cid in ready:
                step_cost = self._cost(cid, baseline)
                future = self._heuristic(remaining - {cid}, baseline)
                # Higher learner relevance is a small secondary incentive; cost
                # remains dominant and dependencies remain absolute.
                f = g + step_cost + future - min(0.25, max(-0.25, float(priority.get(cid, 0)) * 0.02))
                heapq.heappush(heap, (f, step_cost, -float(priority.get(cid, 0)), cid))
            _, step_cost, _, cid = heapq.heappop(heap)
            result.append(cid); done.add(cid); g += step_cost
        return result

def _get(row,key):
    return row[key] if isinstance(row, dict) else getattr(row,key)

def weekly_plan(path_items, id_to_row, weekly_hours=10):
    weekly_hours=max(1.0,float(weekly_hours or 10))
    weeks=[]; week=[]; used=0.0; week_no=1
    for item in path_items:
        if item.get("status") == "completed": continue
        row=id_to_row[item["course_id"]]; remaining=float(_get(row,"duration_hours"))
        while remaining > 0.001:
            capacity=weekly_hours-used
            if capacity <= 0.001:
                weeks.append(_make_week(week_no,week,used)); week_no+=1; week=[]; used=0.0; capacity=weekly_hours
            chunk=min(remaining,capacity)
            week.append({"step":item["step_order"],"course_id":item["course_id"],"title":_get(row,"title"),"skill":_get(row,"skill_tag"),"type":_get(row,"type"),"duration_hours":round(chunk,1),"total_resource_hours":float(_get(row,"duration_hours")),"milestone":item.get("milestone",""),"status":item.get("status","not_started")})
            used += chunk; remaining -= chunk
            if used >= weekly_hours-0.001:
                weeks.append(_make_week(week_no,week,used)); week_no+=1; week=[]; used=0.0
    if week: weeks.append(_make_week(week_no,week,used))
    return weeks

def _make_week(number,items,used): return {"week":number,"hours":round(used,1),"items":items}
def plan_summary(weeks):
    total=sum(w["hours"] for w in weeks)
    return {"weeks":len(weeks),"scheduled_hours":round(total,1),"average_hours":round(total/len(weeks),1) if weeks else 0}
