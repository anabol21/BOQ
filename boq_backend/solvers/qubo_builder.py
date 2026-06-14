from __future__ import annotations

from typing import Any

import pandas as pd


def intervals_overlap(start_a, end_a, start_b, end_b) -> bool:
    return max(start_a, start_b) < min(end_a, end_b)


def _ensure_end_datetime_column(candidate_df: pd.DataFrame) -> pd.DataFrame:
    work_df = candidate_df.copy()
    work_df['show_datetime'] = pd.to_datetime(work_df['show_datetime'])

    if 'end_datetime' in work_df.columns:
        work_df['end_datetime'] = pd.to_datetime(work_df['end_datetime'])
    elif 'show_end_datetime' in work_df.columns:
        work_df['end_datetime'] = pd.to_datetime(work_df['show_end_datetime'])
    elif 'runtime_min' in work_df.columns:
        work_df['end_datetime'] = work_df['show_datetime'] + pd.to_timedelta(work_df['runtime_min'], unit='m')
    else:
        raise KeyError("candidate_df must contain 'end_datetime', 'show_end_datetime', or 'runtime_min'")

    return work_df


def build_conflict_pairs(
    candidate_df: pd.DataFrame,
    cleaning_gap_min: int = 0,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return hall-overlap and cross-hall same-movie conflict pairs."""
    work_df = candidate_df.reset_index().rename(columns={'index': 'candidate_idx'}).copy()
    work_df = _ensure_end_datetime_column(work_df)
    gap = pd.to_timedelta(cleaning_gap_min, unit='m')

    hall_conflict_pairs: list[tuple[int, int]] = []
    same_movie_cross_hall_pairs: list[tuple[int, int]] = []

    rows = list(work_df.itertuples())

    for a_pos in range(len(rows)):
        a = rows[a_pos]
        for b_pos in range(a_pos + 1, len(rows)):
            b = rows[b_pos]

            overlap = intervals_overlap(
                a.show_datetime,
                a.end_datetime + gap,
                b.show_datetime,
                b.end_datetime + gap,
            )
            if not overlap:
                continue

            same_cinema = a.cinema_id == b.cinema_id
            same_hall = a.hall_id == b.hall_id
            same_movie = a.movie_id == b.movie_id

            if same_cinema and same_hall:
                hall_conflict_pairs.append((int(a.candidate_idx), int(b.candidate_idx)))
            elif same_cinema and same_movie and not same_hall:
                same_movie_cross_hall_pairs.append((int(a.candidate_idx), int(b.candidate_idx)))

    return hall_conflict_pairs, same_movie_cross_hall_pairs


def build_qubo_from_candidates(
    candidate_df: pd.DataFrame,
    hall_penalty: float = 1e5,
    same_movie_penalty: float = 2e4,
    cleaning_gap_min: int = 0,
) -> tuple[dict[int, int], list[tuple[int, int, float]], dict[str, Any]]:
    """Build a sparse QUBO matrix from scored candidate shows."""
    work_df = candidate_df.reset_index().rename(columns={'index': 'candidate_idx'}).copy()
    work_df = _ensure_end_datetime_column(work_df)

    var_index_to_candidate_idx: dict[int, int] = {
        var_idx: int(row.candidate_idx)
        for var_idx, row in enumerate(work_df.itertuples())
    }
    candidate_idx_to_var_index: dict[int, int] = {
        cand_idx: var_idx for var_idx, cand_idx in var_index_to_candidate_idx.items()
    }

    q_terms: list[tuple[int, int, float]] = []

    for row in work_df.itertuples():
        var_idx = candidate_idx_to_var_index[int(row.candidate_idx)]
        revenue = float(row.pred_revenue)
        q_terms.append((var_idx, var_idx, -revenue))

    hall_conflict_pairs, same_movie_cross_hall_pairs = build_conflict_pairs(
        work_df,
        cleaning_gap_min=cleaning_gap_min,
    )

    for cand_i, cand_j in hall_conflict_pairs:
        i = candidate_idx_to_var_index[cand_i]
        j = candidate_idx_to_var_index[cand_j]
        if i > j:
            i, j = j, i
        q_terms.append((i, j, float(hall_penalty)))

    for cand_i, cand_j in same_movie_cross_hall_pairs:
        i = candidate_idx_to_var_index[cand_i]
        j = candidate_idx_to_var_index[cand_j]
        if i > j:
            i, j = j, i
        q_terms.append((i, j, float(same_movie_penalty)))

    stats: dict[str, Any] = {
        'n_variables': len(var_index_to_candidate_idx),
        'n_hall_conflicts': len(hall_conflict_pairs),
        'n_same_movie_cross_hall_conflicts': len(same_movie_cross_hall_pairs),
        'hall_penalty': hall_penalty,
        'same_movie_penalty': same_movie_penalty,
        'cleaning_gap_min': cleaning_gap_min,
    }

    return var_index_to_candidate_idx, q_terms, stats


def build_schedule_qubo(problem_data: dict[str, Any]) -> dict[str, Any]:
    """Build QUBO artifacts for the BOQ scheduling problem."""
    candidate_df = problem_data.get('candidate_df')
    if not isinstance(candidate_df, pd.DataFrame):
        raise TypeError("problem_data['candidate_df'] must be a pandas DataFrame")

    var_index_map, q_terms, stats = build_qubo_from_candidates(
        candidate_df,
        hall_penalty=float(problem_data.get('hall_penalty', 1e5)),
        same_movie_penalty=float(problem_data.get('same_movie_penalty', 2e4)),
        cleaning_gap_min=int(problem_data.get('cleaning_gap_min', 0)),
    )

    return {
        'builder': 'qubo',
        'status': 'ok',
        'var_index_to_candidate_idx': var_index_map,
        'quadratic_terms': q_terms,
        'stats': stats,
    }
