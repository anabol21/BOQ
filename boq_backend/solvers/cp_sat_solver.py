from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from ortools.sat.python import cp_model


DEFAULT_CONFIG: dict[str, Any] = {
    'objective_column': 'pred_revenue',
    'time_limit_sec': 10,
    'cleaning_gap_min': 0,
    'allow_movie_repetition': True,
    'max_shows_per_movie': None,
    'min_shows_per_movie': None,
    'prime_time_min_shows_per_movie': None,
    'max_shows_per_hall': None,
    'allowed_formats_by_movie': None,
    'required_format_by_movie': None,
    'hall_capacity_min_by_movie': None,
    'forbidden_pairs': None,
    'penalty_weights': {
        'unselected_slot': 1_000_000,
        'hall_overlap': 1_000_000,
        'movie_format_violation': 50_000,
        'hall_capacity_violation': 50_000,
        'forbidden_pair_violation': 100_000,
        'movie_repetition_violation': 25_000,
        'max_shows_per_movie_violation': 20_000,
        'min_shows_per_movie_violation': 20_000,
        'prime_time_min_shows_violation': 15_000,
        'max_shows_per_hall_violation': 15_000,
        'cross_hall_same_movie_overlap_violation': 30_000,
    },
}


@dataclass(frozen=True)
class CandidateKey:
    idx: int
    show_id: str
    cinema_id: str
    hall_id: str
    movie_id: str


def _normalize_problem_data(problem_data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(DEFAULT_CONFIG)
    normalized.update(problem_data or {})
    penalty_weights = dict(DEFAULT_CONFIG['penalty_weights'])
    penalty_weights.update((problem_data or {}).get('penalty_weights', {}))
    normalized['penalty_weights'] = penalty_weights
    return normalized


def _ensure_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.copy()
    work_df['show_datetime'] = pd.to_datetime(work_df['show_datetime'])

    if 'end_datetime' in work_df.columns:
        work_df['end_datetime'] = pd.to_datetime(work_df['end_datetime'])
    elif 'runtime_min' in work_df.columns:
        work_df['end_datetime'] = work_df['show_datetime'] + pd.to_timedelta(work_df['runtime_min'], unit='m')
    else:
        raise ValueError("candidate_df must contain either 'end_datetime' or 'runtime_min'")

    return work_df


def _validate_candidate_df(candidate_df: pd.DataFrame, objective_column: str) -> None:
    required_columns = {
        'show_id', 'cinema_id', 'hall_id', 'movie_id', 'show_datetime', 'hall_capacity', 'format', objective_column,
    }
    missing = required_columns - set(candidate_df.columns)
    if missing:
        raise ValueError(f"candidate_df is missing required columns: {sorted(missing)}")
    if candidate_df.empty:
        raise ValueError('candidate_df must not be empty')


def _overlap_pairs(df: pd.DataFrame, cleaning_gap_min: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    gap = pd.to_timedelta(cleaning_gap_min, unit='m')

    for _, hall_df in df.groupby(['cinema_id', 'hall_id']):
        hall_df = hall_df.sort_values('show_datetime')
        records = list(hall_df.itertuples())
        for i in range(len(records)):
            left = records[i]
            left_end = left.end_datetime + gap
            for j in range(i + 1, len(records)):
                right = records[j]
                if right.show_datetime < left_end and left.show_datetime < right.end_datetime + gap:
                    pairs.append((int(left.Index), int(right.Index)))
    return pairs


def _cross_hall_same_movie_overlap_groups(
    df: pd.DataFrame,
    cleaning_gap_min: int = 0,
) -> list[dict[str, Any]]:
    """Return connected groups of same-movie shows that overlap across different halls."""
    gap = pd.to_timedelta(cleaning_gap_min, unit='m')
    groups: list[dict[str, Any]] = []

    for (cinema_id, movie_id), movie_df in df.groupby(['cinema_id', 'movie_id']):
        records = list(movie_df.itertuples())
        if len(records) < 2:
            continue

        adjacency: dict[int, set[int]] = {i: set() for i in range(len(records))}
        for i in range(len(records)):
            left = records[i]
            left_end = left.end_datetime + gap
            for j in range(i + 1, len(records)):
                right = records[j]
                if left.hall_id == right.hall_id:
                    continue
                if right.show_datetime < left_end and left.show_datetime < right.end_datetime + gap:
                    adjacency[i].add(j)
                    adjacency[j].add(i)

        visited: set[int] = set()
        for start in range(len(records)):
            if start in visited or not adjacency[start]:
                continue
            component: list[int] = []
            stack = [start]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                stack.extend(neighbor for neighbor in adjacency[node] if neighbor not in visited)

            if len(component) < 2:
                continue

            indices = [int(records[i].Index) for i in component]
            hall_ids = sorted({str(records[i].hall_id) for i in component})
            groups.append({
                'cinema_id': str(cinema_id),
                'movie_id': str(movie_id),
                'hall_ids': hall_ids,
                'indices': indices,
            })

    return groups


def _format_allowed(row: pd.Series, config: dict[str, Any]) -> bool:
    allowed_map = config.get('allowed_formats_by_movie') or {}
    required_map = config.get('required_format_by_movie') or {}
    movie_id = row['movie_id']
    row_format = row['format']

    if movie_id in required_map:
        return row_format == required_map[movie_id]
    if movie_id in allowed_map:
        return row_format in set(allowed_map[movie_id])
    return True


def _capacity_allowed(row: pd.Series, config: dict[str, Any]) -> bool:
    min_capacity_map = config.get('hall_capacity_min_by_movie') or {}
    movie_id = row['movie_id']
    required_capacity = min_capacity_map.get(movie_id)
    if required_capacity is None:
        return True
    return int(row['hall_capacity']) >= int(required_capacity)


def _forbidden_pair_set(config: dict[str, Any]) -> set[tuple[str, str]]:
    raw_pairs = config.get('forbidden_pairs') or []
    normalized = set()
    for left, right in raw_pairs:
        normalized.add(tuple(sorted((str(left), str(right)))))
    return normalized


def solve_cp_sat_schedule(problem_data: dict[str, Any]) -> dict[str, Any]:
    """Universal CP-SAT baseline for BOQ cinema scheduling.

    Expected problem_data keys:
    - candidate_df: pandas DataFrame with one row per feasible-or-nearly-feasible candidate show.
      Recommended columns: show_id, cinema_id, hall_id, movie_id, show_datetime, runtime_min or end_datetime,
      hall_capacity, format, slot_id (optional), prime_time_flag (optional), and objective column.
    - objective_column: score to maximize, default 'pred_revenue'.
    - cleaning_gap_min: turnaround gap added between consecutive shows in the same hall.
    - allow_movie_repetition: if False, each movie may be selected at most once.
    - max_shows_per_movie/min_shows_per_movie: hard counts unless soft_constraints=True.
    - prime_time_min_shows_per_movie: minimum prime-time selections per movie.
    - max_shows_per_hall: upper bound on chosen shows per hall.
    - allowed_formats_by_movie / required_format_by_movie: movie-to-format compatibility rules.
    - hall_capacity_min_by_movie: minimum hall capacity required by movie.
    - forbidden_pairs: list of movie pairs that cannot both be selected.
    - soft_constraints: when True, impossible scenarios are represented as penalty slack variables.

    The formulation is intentionally richer than the toy notebook case so the MVP can demonstrate
    both hard feasibility logic and penalty-based modeling of impossible scenarios.
    """
    config = _normalize_problem_data(problem_data)
    candidate_df = config.get('candidate_df')
    if not isinstance(candidate_df, pd.DataFrame):
        raise TypeError("problem_data['candidate_df'] must be a pandas DataFrame")

    objective_column = str(config['objective_column'])
    soft_constraints = bool(config.get('soft_constraints', False))
    time_limit_sec = float(config['time_limit_sec'])

    _validate_candidate_df(candidate_df, objective_column)
    work_df = _ensure_datetime_columns(candidate_df).reset_index(drop=True)
    work_df[objective_column] = pd.to_numeric(work_df[objective_column], errors='raise')
    work_df['objective_int'] = work_df[objective_column].round().astype(int)
    if 'slot_id' not in work_df.columns:
        work_df['slot_id'] = work_df['show_datetime'].dt.strftime('%Y%m%d_%H%M')
    if 'prime_time_flag' not in work_df.columns:
        work_df['prime_time_flag'] = False

    model = cp_model.CpModel()
    x: dict[int, cp_model.IntVar] = {}
    for idx, row in work_df.iterrows():
        x[idx] = model.NewBoolVar(f"x_{idx}_{row['hall_id']}_{row['movie_id']}")

    penalties: list[cp_model.LinearExpr] = []
    penalty_details: dict[str, list[dict[str, Any]]] = {key: [] for key in config['penalty_weights']}

    def add_at_most_one(indices: list[int], violation_name: str, context: dict[str, Any]) -> None:
        if not indices:
            return
        if soft_constraints:
            slack = model.NewIntVar(0, len(indices), f"slack_{violation_name}_{len(penalty_details[violation_name])}")
            model.Add(sum(x[i] for i in indices) <= 1 + slack)
            penalties.append(config['penalty_weights'][violation_name] * slack)
            penalty_details[violation_name].append(context | {'slack_var': slack.Name()})
        else:
            model.Add(sum(x[i] for i in indices) <= 1)

    def add_exactly_one(indices: list[int], violation_name: str, context: dict[str, Any]) -> None:
        if not indices:
            return
        if soft_constraints:
            shortfall = model.NewIntVar(0, 1, f"shortfall_{violation_name}_{len(penalty_details[violation_name])}")
            excess = model.NewIntVar(0, len(indices), f"excess_{violation_name}_{len(penalty_details[violation_name])}")
            model.Add(sum(x[i] for i in indices) + shortfall - excess == 1)
            penalties.append(config['penalty_weights'][violation_name] * (shortfall + excess))
            penalty_details[violation_name].append(context | {'shortfall_var': shortfall.Name(), 'excess_var': excess.Name()})
        else:
            model.Add(sum(x[i] for i in indices) == 1)

    for slot_id, indices in work_df.groupby('slot_id').groups.items():
        add_exactly_one(list(indices), 'unselected_slot', {'slot_id': slot_id})

    for left_idx, right_idx in _overlap_pairs(work_df, int(config['cleaning_gap_min'])):
        # Physical feasibility in the same hall must remain hard even in "soft" runs:
        # two overlapping candidates in the same hall cannot both be selected.
        model.Add(x[left_idx] + x[right_idx] <= 1)
        if soft_constraints:
            penalty_details['hall_overlap'].append(
                {
                    'left_show_id': work_df.loc[left_idx, 'show_id'],
                    'right_show_id': work_df.loc[right_idx, 'show_id'],
                    'hall_id': work_df.loc[left_idx, 'hall_id'],
                    'cinema_id': work_df.loc[left_idx, 'cinema_id'],
                    'enforced_as_hard': True,
                }
            )

    invalid_format_indices = [idx for idx, row in work_df.iterrows() if not _format_allowed(row, config)]
    for idx in invalid_format_indices:
        if soft_constraints:
            penalties.append(config['penalty_weights']['movie_format_violation'] * x[idx])
            penalty_details['movie_format_violation'].append({'show_id': work_df.loc[idx, 'show_id']})
        else:
            model.Add(x[idx] == 0)

    invalid_capacity_indices = [idx for idx, row in work_df.iterrows() if not _capacity_allowed(row, config)]
    for idx in invalid_capacity_indices:
        if soft_constraints:
            penalties.append(config['penalty_weights']['hall_capacity_violation'] * x[idx])
            penalty_details['hall_capacity_violation'].append({'show_id': work_df.loc[idx, 'show_id']})
        else:
            model.Add(x[idx] == 0)

    forbidden_pairs = _forbidden_pair_set(config)
    if forbidden_pairs:
        by_movie = {movie_id: list(indices) for movie_id, indices in work_df.groupby('movie_id').groups.items()}
        for left_movie, right_movie in forbidden_pairs:
            left_indices = by_movie.get(left_movie, [])
            right_indices = by_movie.get(right_movie, [])
            if not left_indices or not right_indices:
                continue
            if soft_constraints:
                slack = model.NewIntVar(0, len(left_indices) + len(right_indices), f"slack_forbidden_{left_movie}_{right_movie}")
                model.Add(sum(x[i] for i in left_indices) + sum(x[i] for i in right_indices) <= 1 + slack)
                penalties.append(config['penalty_weights']['forbidden_pair_violation'] * slack)
                penalty_details['forbidden_pair_violation'].append({'movie_pair': [left_movie, right_movie], 'slack_var': slack.Name()})
            else:
                model.Add(sum(x[i] for i in left_indices) + sum(x[i] for i in right_indices) <= 1)

    by_movie = {movie_id: list(indices) for movie_id, indices in work_df.groupby('movie_id').groups.items()}
    if not bool(config['allow_movie_repetition']):
        for movie_id, indices in by_movie.items():
            add_at_most_one(indices, 'movie_repetition_violation', {'movie_id': movie_id})

    max_shows_per_movie = config.get('max_shows_per_movie')
    if max_shows_per_movie is not None:
        for movie_id, indices in by_movie.items():
            if soft_constraints:
                slack = model.NewIntVar(0, len(indices), f"slack_max_movie_{movie_id}")
                model.Add(sum(x[i] for i in indices) <= int(max_shows_per_movie) + slack)
                penalties.append(config['penalty_weights']['max_shows_per_movie_violation'] * slack)
                penalty_details['max_shows_per_movie_violation'].append({'movie_id': movie_id, 'slack_var': slack.Name()})
            else:
                model.Add(sum(x[i] for i in indices) <= int(max_shows_per_movie))

    min_shows_per_movie = config.get('min_shows_per_movie')
    if min_shows_per_movie is not None:
        for movie_id, indices in by_movie.items():
            if soft_constraints:
                slack = model.NewIntVar(0, len(indices), f"slack_min_movie_{movie_id}")
                model.Add(sum(x[i] for i in indices) + slack >= int(min_shows_per_movie))
                penalties.append(config['penalty_weights']['min_shows_per_movie_violation'] * slack)
                penalty_details['min_shows_per_movie_violation'].append({'movie_id': movie_id, 'slack_var': slack.Name()})
            else:
                model.Add(sum(x[i] for i in indices) >= int(min_shows_per_movie))

    prime_time_min = config.get('prime_time_min_shows_per_movie')
    if prime_time_min is not None:
        for movie_id, indices in by_movie.items():
            prime_indices = [i for i in indices if bool(work_df.loc[i, 'prime_time_flag'])]
            if not prime_indices:
                continue
            if soft_constraints:
                slack = model.NewIntVar(0, len(prime_indices), f"slack_prime_{movie_id}")
                model.Add(sum(x[i] for i in prime_indices) + slack >= int(prime_time_min))
                penalties.append(config['penalty_weights']['prime_time_min_shows_violation'] * slack)
                penalty_details['prime_time_min_shows_violation'].append({'movie_id': movie_id, 'slack_var': slack.Name()})
            else:
                model.Add(sum(x[i] for i in prime_indices) >= int(prime_time_min))

    max_shows_per_hall = config.get('max_shows_per_hall')
    if max_shows_per_hall is not None:
        for hall_key, indices in work_df.groupby(['cinema_id', 'hall_id']).groups.items():
            cinema_id, hall_id = hall_key
            if soft_constraints:
                slack = model.NewIntVar(0, len(indices), f"slack_hall_{cinema_id}_{hall_id}")
                model.Add(sum(x[i] for i in indices) <= int(max_shows_per_hall) + slack)
                penalties.append(config['penalty_weights']['max_shows_per_hall_violation'] * slack)
                penalty_details['max_shows_per_hall_violation'].append({'cinema_id': cinema_id, 'hall_id': hall_id, 'slack_var': slack.Name()})
            else:
                model.Add(sum(x[i] for i in indices) <= int(max_shows_per_hall))

    cross_hall_same_movie_penalty = config.get('cross_hall_same_movie_penalty')
    overlap_groups = _cross_hall_same_movie_overlap_groups(work_df, int(config['cleaning_gap_min']))
    if overlap_groups:
        penalty_weight = int(
            config['penalty_weights'].get(
                'cross_hall_same_movie_overlap_violation',
                cross_hall_same_movie_penalty if cross_hall_same_movie_penalty is not None else 30000,
            )
        )
        for group in overlap_groups:
            indices = group['indices']
            if soft_constraints:
                slack = model.NewIntVar(0, len(indices), f"slack_cross_hall_same_movie_{group['movie_id']}_{len(indices)}")
                model.Add(sum(x[i] for i in indices) <= 1 + slack)
                penalties.append(penalty_weight * slack)
                penalty_details['cross_hall_same_movie_overlap_violation'].append({
                    'cinema_id': group['cinema_id'],
                    'movie_id': group['movie_id'],
                    'hall_ids': group['hall_ids'],
                    'slack_var': slack.Name(),
                })
            else:
                model.Add(sum(x[i] for i in indices) <= 1)

    objective_expr = sum(work_df.loc[idx, 'objective_int'] * x[idx] for idx in work_df.index)
    if penalties:
        model.Maximize(objective_expr - sum(penalties))
    else:
        model.Maximize(objective_expr)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_search_workers = 8
    status_code = solver.Solve(model)
    status_name = solver.StatusName(status_code)

    selected_indices = [idx for idx in work_df.index if solver.Value(x[idx]) == 1]
    selected_df = (
        work_df.loc[selected_indices]
        .copy()
        .sort_values(['cinema_id', 'hall_id', 'show_datetime', 'movie_id'])
        .reset_index(drop=True)
    )

    selected_movie_ids_by_slot: dict[str, str] = {}
    for _, row in selected_df.iterrows():
        selected_movie_ids_by_slot[str(row['slot_id'])] = str(row['movie_id'])

    hall_schedules: dict[str, list[dict[str, Any]]] = {}
    for (cinema_id, hall_id), hall_df in selected_df.groupby(['cinema_id', 'hall_id']):
        hall_schedules[f'{cinema_id}::{hall_id}'] = hall_df[
            ['show_id', 'slot_id', 'movie_id', 'show_datetime', 'end_datetime', objective_column]
        ].to_dict(orient='records')

    active_penalties: dict[str, Any] = {}
    for penalty_name, details in penalty_details.items():
        active_penalties[penalty_name] = details

    overlap_groups_summary = _cross_hall_same_movie_overlap_groups(work_df, int(config['cleaning_gap_min']))

    hard_conflicts = {
        'overlap_pairs_in_candidate_set': _overlap_pairs(work_df, int(config['cleaning_gap_min'])),
        'invalid_format_candidate_count': len(invalid_format_indices),
        'invalid_capacity_candidate_count': len(invalid_capacity_indices),
    }

    return {
        'solver': 'cp_sat',
        'status': status_name,
        'objective_column': objective_column,
        'objective_value': float(selected_df[objective_column].sum()) if not selected_df.empty else 0.0,
        'selected_schedule_df': selected_df,
        'selected_movie_ids_by_slot': selected_movie_ids_by_slot,
        'hall_schedules': hall_schedules,
        'num_candidates': int(len(work_df)),
        'num_selected': int(len(selected_df)),
        'num_slots': int(work_df['slot_id'].nunique()),
        'num_halls': int(work_df[['cinema_id', 'hall_id']].drop_duplicates().shape[0]),
        'wall_time_sec': float(solver.WallTime()),
        'best_objective_bound': float(solver.BestObjectiveBound()),
        'soft_constraints': soft_constraints,
        'config': config,
        'hard_conflicts': hard_conflicts,
        'cross_hall_same_movie_overlap_groups': overlap_groups_summary,
        'penalty_details': active_penalties,
    }
