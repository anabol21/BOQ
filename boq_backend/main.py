from __future__ import annotations

from boq_backend.solvers import build_schedule_qubo, solve_cp_sat_schedule, solve_qubo_annealing



def demo() -> None:
    sample_problem = {
        "candidate_shows": [],
        "constraints": {},
        "objective": "expected_attendance",
    }

    cp_sat_result = solve_cp_sat_schedule(sample_problem)
    qubo = build_schedule_qubo(sample_problem)
    anneal_result = solve_qubo_annealing(qubo)

    print(cp_sat_result)
    print(qubo)
    print(anneal_result)


if __name__ == "__main__":
    demo()
