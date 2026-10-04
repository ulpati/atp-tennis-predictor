# Methodology

This page explains how features are built and how the model is evaluated.

## The leakage trap

Match predictors usually emit two rows per match, one per player, with the
winner's row labelled 1. If a single column is computed from the winner's
point of view and copied into both rows, for example "winner's ranking
points minus loser's", the model can tell the two rows apart and its
accuracy climbs above 80% without predicting anything. This project rules
it out by construction: both rows come from the same function with the
players swapped, and the tests check it on every match.

## Features

All features describe the situation **before** the match. The builder walks
through the matches in chronological order. Matches of the same tournament
share a start date, so they are ordered by round. For each match it:

1. records public pre-match information (ranking, ranking points, age, height, hand);
2. reads a snapshot of both players and of their matchup;
3. only then applies the result to ratings, records, statistics and the head-to-head.

Prediction reads snapshots from the final state with the same functions,
so the vectors used in training and in prediction are identical by
construction. A test rebuilds the state up to a match and checks that the
prediction-time vector equals the stored training row.

| Group | Per player (p1, p2 and their difference) |
|---|---|
| Ratings | overall Elo, surface Elo (dynamic K: `250 / (matches + 5) ** 0.4`) |
| Ranking | log of ATP rank, log of ranking points |
| Profile | age, height, left-handed, log of matches played |
| Results | smoothed career and surface win rate, recent form, momentum, form variability, current streak |
| Serve and return | serve points won, return points won, ace and double-fault rate, first-serve percentage, break points saved, over the last 30 matches with statistics |
| Workload | days since the last match, matches and minutes in the previous 14 days |

Matchup features: Elo win probability (overall and surface), head-to-head
wins of each player and their difference, on all surfaces and on this one.
Context: surface and whether the match is best of five.

Every match yields two mirrored rows from the same function with the
players swapped. Tests verify on every row that the second row is the exact
mirror of the first, and that changing who won a match does not change any
feature of that match or of earlier matches.

Walkovers and defaults are removed because no tennis was played.

## Evaluation protocol

- **Split by date.** Train, validation and test periods are contiguous and a
  tournament start date never spans two periods. By default the test period
  is the last 10% of matches and the validation period the 10% before it.
- **Decisions on validation only.** Early stopping, random search,
  isotonic calibration and the choice between global and per-surface models
  use the validation period. The test period is used once, for reporting.
- **Per-match metrics.** For each test match the reported probability is
  `(p(A beats B) + 1 - p(B beats A)) / 2`, exactly what `predict` returns.
  Accuracy, log loss, Brier score and AUC are computed on it.
- **Baselines** on the same matches: overall Elo, surface Elo and "the
  better-ranked player wins".
- **Cross-validation** (`--cv-folds`) uses expanding windows on the
  pre-test period; each fold early-stops on the tail of its own training data.

## Final model

Once the test metrics are computed, `atp train` retrains each chosen model
with the same parameters and the number of trees found by early stopping on
every match, test period included, and saves that model for predictions.
Nothing is chosen in this step, so the test period still plays no part in
any decision. The reported metrics describe the evaluated model, which never
saw the test period; they estimate how the procedure performs on new
matches. `--no-refit` saves the evaluated model instead.

Isotonic calibration is learned on the validation period, which the final
model trains on, so `--calibration isotonic` requires `--no-refit`.

## Results

Produced by `atp download` and `atp train` with the default options. Data:
197,918 tour-level matches from 1968 to 25 May 2026. Matches before 1970 only
warm up ratings and statistics.

| Period | Matches | Dates |
|---|---|---|
| Train | 152,466 | 1970-01-04 to 2012-08-27 |
| Validation | 19,068 | 2012-09-14 to 2019-05-06 |
| Test | 19,086 | 2019-05-13 to 2026-05-25 |

| Predictor (test period) | Accuracy | Log loss | Brier | AUC |
|---|---|---|---|---|
| XGBoost model | 66.3% | 0.603 | 0.209 | 0.731 |
| Elo only | 64.3% | 0.640 | 0.223 | 0.702 |
| Surface Elo only | 63.7% | 0.648 | 0.226 | 0.694 |
| Better ranking wins | 63.9% | | | |

Five-fold cross-validation on the earlier periods gives 69.5% ± 1.9%
accuracy and a log loss of 0.575. Recent seasons are harder to predict than
the years dominated by a few players. The final model is then retrained on
all 190,620 matches from 1970 with 881 trees.

Options that did **not** help on this data, measured on the test period
after being chosen on validation:

| Option | Test log loss |
|---|---|
| Defaults | 0.6033 |
| `--per-surface --calibration isotonic --no-refit` | 0.6047 |
| `--search 12` | 0.6034 |

The per-surface comparison kept the global model for hard and clay courts
and chose a grass model, and calibration raised the log loss from 0.6032 to
0.6047. The best of the 12 random-search candidates only tied the defaults on
validation (0.5795), so the default parameters were not changed.

### Choice of the data range

On the validation period the range of years makes no measurable difference:

| Files | Training rows from | Validation accuracy | Validation log loss |
|---|---|---|---|
| 1985 onwards | 1991 | 69.07% | 0.57912 |
| 1968 onwards | 1991 | 69.06% | 0.57945 |
| 1968 onwards | 1981 | 68.85% | 0.57965 |
| 1968 onwards (default) | 1970 | 69.07% | 0.57926 |

The defaults use every file. Ratings and head-to-head records of players
whose careers began before 1985 are then complete, and the commands need no
options.

## Limitations

- The data ends with Roland Garros 2026, because the original repository is
  no longer updated. Ratings and form are frozen at that date.
- No information about injuries, withdrawals, odds or live conditions.
- Players with few matches have unreliable ratings; `predict` warns about them.
- Predictions use each player's latest state. A date earlier than the end of
  the data is refused, because the state cannot be rewound.
- Accuracy around 66% is in line with published ATP models. Betting markets
  are usually a few points better, so this tool is not suitable for betting.
