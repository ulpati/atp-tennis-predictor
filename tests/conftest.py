"""Shared fixtures: a synthetic ATP dataset in Jeff Sackmann's CSV format."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atp_core.cli import main

COLUMNS = [
    "tourney_id",
    "tourney_name",
    "surface",
    "draw_size",
    "tourney_level",
    "tourney_date",
    "match_num",
    "winner_id",
    "winner_seed",
    "winner_entry",
    "winner_name",
    "winner_hand",
    "winner_ht",
    "winner_ioc",
    "winner_age",
    "loser_id",
    "loser_seed",
    "loser_entry",
    "loser_name",
    "loser_hand",
    "loser_ht",
    "loser_ioc",
    "loser_age",
    "score",
    "best_of",
    "round",
    "minutes",
    "w_ace",
    "w_df",
    "w_svpt",
    "w_1stIn",
    "w_1stWon",
    "w_2ndWon",
    "w_SvGms",
    "w_bpSaved",
    "w_bpFaced",
    "l_ace",
    "l_df",
    "l_svpt",
    "l_1stIn",
    "l_1stWon",
    "l_2ndWon",
    "l_SvGms",
    "l_bpSaved",
    "l_bpFaced",
    "winner_rank",
    "winner_rank_points",
    "loser_rank",
    "loser_rank_points",
]

FIRST = [
    "Novak",
    "Carlos",
    "Jannik",
    "Daniil",
    "Andrey",
    "Casper",
    "Holger",
    "Taylor",
    "Stefanos",
    "Alex",
    "Hubert",
    "Grigor",
    "Tommy",
    "Frances",
    "Karen",
    "Lorenzo",
    "Ugo",
    "Ben",
    "Sebastian",
    "Felix",
    "Nicolas",
    "Alejandro",
    "Cameron",
    "Adrian",
    "Tallon",
    "Matteo",
    "Jiri",
    "Arthur",
    "Luca",
    "Ana",
    "Roman",
    "Jack",
    "Mariano",
    "Botic",
    "Christopher",
    "Jordan",
    "Marcos",
    "Gael",
    "Pedro",
    "José",
]
LAST = [
    "Djokovic",
    "Alcaraz",
    "Sinner",
    "Medvedev",
    "Rublev",
    "Ruud",
    "Rune",
    "Fritz",
    "Tsitsipas",
    "Minaur",
    "Hurkacz",
    "Dimitrov",
    "Paul",
    "Tiafoe",
    "Khachanov",
    "Musetti",
    "Humbert",
    "Shelton",
    "Korda",
    "Auger",
    "Jarry",
    "Fokina",
    "Norrie",
    "Mannarino",
    "Griekspoor",
    "Arnaldi",
    "Lehecka",
    "Fils",
    "Rossi",
    "Rossi",
    "Safiullin",
    "Draper",
    "Navone",
    "Zandschulp",
    "Eubanks",
    "Thompson",
    "Giron",
    "Monfils",
    "Martinez",
    "Álvarez",
]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def make_matches(first_year: int = 2014, last_year: int = 2019, seed: int = 7) -> pd.DataFrame:
    """Single-elimination tournaments whose winners depend on a latent skill per player."""
    rng = np.random.default_rng(seed)
    n_players = len(FIRST)
    skill = rng.normal(0.0, 1.0, n_players)
    clay_bonus = rng.normal(0.0, 0.4, n_players)
    hand = rng.choice(["R", "R", "R", "L"], n_players)
    height = rng.integers(175, 205, n_players)
    birth_age = rng.uniform(18, 30, n_players)
    rows = []
    match_num = 0
    for year in range(first_year, last_year + 1):
        for week in range(0, 50, 2):
            date = pd.Timestamp(year=year, month=1, day=6) + pd.Timedelta(weeks=week)
            surface = rng.choice(["Hard", "Clay", "Grass"], p=[0.5, 0.35, 0.15])
            level, best_of = ("G", 5) if week % 12 == 0 else ("A", 3)
            draw = list(rng.choice(n_players, 16, replace=False))
            noisy = skill + rng.normal(0, 0.3, n_players)
            rank_of = {p: r + 1 for r, p in enumerate(np.argsort(-noisy))}
            tid = f"{year}-{week:03d}"
            for rnd in ("R16", "QF", "SF", "F"):
                next_draw = []
                for a, b in zip(draw[0::2], draw[1::2], strict=True):
                    edge = 1.3 * (skill[a] - skill[b]) + (clay_bonus[a] - clay_bonus[b]) * (surface == "Clay")
                    winner, loser = (a, b) if rng.random() < _sigmoid(edge) else (b, a)
                    next_draw.append(winner)
                    match_num += 1
                    rows.append(
                        _row(
                            tid,
                            surface,
                            level,
                            best_of,
                            date,
                            match_num,
                            rnd,
                            winner,
                            loser,
                            rank_of,
                            hand,
                            height,
                            birth_age,
                            year - first_year,
                            rng,
                        )
                    )
                draw = next_draw
    df = pd.DataFrame(rows, columns=COLUMNS)
    walkover = df.iloc[[5]].copy()
    walkover["score"] = "W/O"
    no_surface = df.iloc[[6]].copy()
    no_surface["surface"] = np.nan
    no_surface["match_num"] = 99999
    return pd.concat([df, walkover, no_surface], ignore_index=True)


def _row(tid, surface, level, best_of, date, match_num, rnd, w, lo, rank_of, hand, height, birth_age, years, rng):
    def serve(better: bool):
        svpt = int(rng.integers(55, 95))
        first_in = int(svpt * rng.uniform(0.55, 0.68))
        first_won = int(first_in * rng.uniform(0.68, 0.80) + (3 if better else 0))
        second_won = int((svpt - first_in) * rng.uniform(0.45, 0.58))
        bp_faced = int(rng.integers(1, 10))
        return [
            int(rng.integers(0, 15)),
            int(rng.integers(0, 6)),
            svpt,
            first_in,
            min(first_won, first_in),
            second_won,
            int(svpt / 6.5),
            int(rng.integers(0, bp_faced + 1)),
            bp_faced,
        ]

    return [
        tid,
        f"Synthetic Open {tid}",
        surface,
        16,
        level,
        date.strftime("%Y%m%d"),
        match_num,
        str(1000 + w),
        "",
        "",
        f"{FIRST[w]} {LAST[w]}",
        hand[w],
        height[w],
        "ITA",
        round(birth_age[w] + years, 1),
        str(1000 + lo),
        "",
        "",
        f"{FIRST[lo]} {LAST[lo]}",
        hand[lo],
        height[lo],
        "ITA",
        round(birth_age[lo] + years, 1),
        "6-4 6-4",
        best_of,
        rnd,
        int(rng.integers(60, 200)),
        *serve(True),
        *serve(False),
        rank_of[w],
        10000 // rank_of[w],
        rank_of[lo],
        10000 // rank_of[lo],
    ]


def write_csvs(df: pd.DataFrame, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    years = df["tourney_date"].astype(str).str[:4]
    for year, part in df.groupby(years):
        part.to_csv(directory / f"atp_matches_{year}.csv", index=False)
    return directory


@pytest.fixture(scope="session")
def raw_matches() -> pd.DataFrame:
    return make_matches()


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory: pytest.TempPathFactory, raw_matches: pd.DataFrame) -> Path:
    return write_csvs(raw_matches, tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="session")
def trained_models_dir(tmp_path_factory: pytest.TempPathFactory, data_dir: Path) -> Path:
    """One trained run shared by the read-only tests."""
    models = tmp_path_factory.mktemp("models")
    code = main(
        [
            "--log-level",
            "WARNING",
            "train",
            "--data-dir",
            str(data_dir),
            "--models-dir",
            str(models),
            "--max-rounds",
            "80",
            "--threads",
            "2",
        ]
    )
    assert code == 0
    return models
