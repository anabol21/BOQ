# BOQ MVP Dataset Schema

## Purpose

This document defines the first MVP dataset schema for BOQ: a hybrid schedule optimization service for cinema scheduling. The first stage uses a fully classical baseline for demand prediction and schedule optimization; later stages extend the same pipeline with QUBO formulations and quantum-inspired solvers.

The dataset is designed at the **show level**:
- One row = one movie show in one hall at one specific datetime.
- Primary target = `occupancy_rate = sold_tickets / hall_capacity`.
- Secondary target = `sold_tickets`.

## Core principles

- Prediction target should be available for historical supervised learning.
- Features used for planning-time prediction must avoid leakage.
- Validation split should be time-based: earlier periods for train, later periods for validation.
- The same schema should support both forecasting and downstream schedule optimization.

## Required fields

| Field | Type | Role | Description |
|---|---|---|---|
| `show_id` | string | feature/id | Unique show identifier. |
| `cinema_id` | string | feature/id | Unique cinema identifier. |
| `hall_id` | string | feature/id | Unique hall identifier inside a cinema. |
| `hall_capacity` | int | feature | Total number of seats in the hall. |
| `movie_id` | string | feature/id | Unique movie identifier. |
| `genre` | category | feature | Main genre or normalized genre label. |
| `age_rating` | category | feature | Age restriction category. |
| `runtime_min` | int | feature | Movie runtime in minutes. |
| `release_date` | date | feature | Official release date of the movie. |
| `show_datetime` | datetime | feature | Start datetime of the show. |
| `format` | category | feature | Screening format, e.g. `2D`, `3D`, `IMAX`. |
| `base_price` | float | feature | Base ticket price known at planning time. |
| `sold_tickets` | int | target | Number of sold tickets for the show. |
| `occupancy_rate` | float | target | `sold_tickets / hall_capacity`, clipped to `[0, 1]` for clean data. |

## Derived features for MVP

| Field | Type | Description |
|---|---|---|
| `city` | category | City of the cinema. |
| `day_of_week` | int/category | Day of week derived from `show_datetime`. |
| `is_weekend` | bool | Weekend indicator. |
| `hour_bucket` | category | Time-of-day bucket, e.g. morning/day/evening/late. |
| `month` | int | Month extracted from `show_datetime`. |
| `holiday_flag` | bool | Holiday indicator from a selected holiday calendar. |
| `release_age_days` | int | Days between `show_datetime` and `release_date`. |
| `lead_time_days` | float | Days between publication and show time, if publish time exists. |
| `prime_time_flag` | bool | Whether the show is in a business-defined prime-time window. |
| `runtime_bucket` | category | Bucketized runtime. |
| `screening_number_for_movie` | int | Sequential screening count for a movie in a local context. |

## Leakage guardrails

For the baseline long-horizon predictor, do **not** use:
- Final revenue.
- Post-factum ratings or signals unavailable at planning time.
- Sales snapshots very close to show start if the prediction is meant for earlier planning.
- Any feature unavailable at the actual schedule-construction decision time.

## Split strategy

Use a **time-based split**:
1. Earlier weeks/months -> training.
2. Later weeks/months -> validation.
3. Optionally keep the latest slice as a test set.

Avoid random row-wise splitting because it can leak temporal demand structure.

## Data quality checks

Recommended validation rules:
- `hall_capacity > 0`
- `sold_tickets >= 0`
- `sold_tickets <= hall_capacity` for clean finalized rows
- `0 <= occupancy_rate <= 1`
- `release_date <= show_datetime`
- required IDs must be non-empty
- categorical fields should be normalized to canonical labels

## Modeling notes

For the first MVP stage:
- Predict `occupancy_rate` as the primary target.
- Keep `sold_tickets` as a parallel target for alternative objective definitions.
- Start with simple baselines: mean encoders, tree-based regressors, and strong time/calendar features.
- Feed predicted demand into schedule optimization later.

## Optimization handoff

The schedule optimizer can later consume:
- predicted occupancy or sold tickets,
- hall capacities,
- show timing constraints,
- format compatibility,
- business objectives such as utilization, expected attendance, and diversity of programming.
