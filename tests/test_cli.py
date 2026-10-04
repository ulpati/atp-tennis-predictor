import csv
import json
import os
import subprocess
import sys

import pytest

from atp_core import cli
from atp_core.cli import main


def run(capsys, *args):
    code = main(["--log-level", "ERROR", *args])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_version(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "atp 1." in capsys.readouterr().out


def test_evaluate_table_and_json(trained_models_dir, capsys):
    code, out, _ = run(capsys, "evaluate", "--models-dir", str(trained_models_dir))
    assert code == 0 and "never used for training or tuning" in out and "Better ranking wins" in out
    code, out, _ = run(capsys, "evaluate", "--models-dir", str(trained_models_dir), "--json")
    data = json.loads(out)
    assert code == 0 and data["metrics"]["test"]["matches"] > 0


def test_predict_table_json_and_explain(trained_models_dir, capsys):
    code, out, _ = run(
        capsys,
        "predict",
        "Jannik Sinner",
        "Carlos Alcaraz",
        "hard",
        "--models-dir",
        str(trained_models_dir),
        "--explain",
    )
    assert code == 0 and "Predicted winner" in out and "Main factors" in out
    code, out, _ = run(
        capsys,
        "predict",
        "Sinner",
        "Alcaraz",
        "Grass",
        "--best-of",
        "5",
        "--json",
        "--models-dir",
        str(trained_models_dir),
    )
    assert code == 0 and json.loads(out)["best_of"] == 5


def test_predict_unknown_player_is_a_clean_error(trained_models_dir, capsys):
    code, _, err = run(
        capsys, "predict", "Nobody Known", "Carlos Alcaraz", "Hard", "--models-dir", str(trained_models_dir)
    )
    assert code == 1 and "not found" in err and "Traceback" not in err


def test_invalid_arguments_exit_with_usage_error(capsys):
    for args in (
        ["predict", "a", "b", "Sand"],
        ["predict", "a", "b", "Hard", "--date", "yesterday"],
        ["train", "--test-fraction", "0.9"],
        ["train", "--seed", "-1"],
        ["train", "--seed", str(2**31)],
        ["list", "--top", "0"],
    ):
        with pytest.raises(SystemExit) as excinfo:
            main(args)
        assert excinfo.value.code == 2
    capsys.readouterr()


def test_batch_with_header_errors_and_formula_injection(trained_models_dir, tmp_path, capsys):
    batch = tmp_path / "matches.csv"
    batch.write_text(
        "player1,player2,surface\n"
        "# a comment\n"
        '"Sinner, Jannik",Carlos Alcaraz,Hard\n'
        "Jannik Sinner,Carlos Alcaraz,Clay,5\n"
        "=cmd|' /C calc'!A0,Carlos Alcaraz,Hard\n",
        encoding="utf-8",
    )
    output = tmp_path / "out.csv"
    code, out, _ = run(capsys, "batch", str(batch), "--output", str(output), "--models-dir", str(trained_models_dir))
    assert code == 1  # two rows cannot be resolved
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert len(rows) == 3
    assert rows[1]["best_of"] == "5" and rows[1]["predicted_winner"]
    assert rows[2]["player_a"].startswith("'=")  # neutralised for spreadsheets
    code, out, _ = run(capsys, "batch", str(batch), "--json", "--models-dir", str(trained_models_dir))
    assert len(json.loads(out)) == 3


def test_batch_reports_warnings(trained_models_dir, tmp_path, capsys):
    batch = tmp_path / "later.csv"
    batch.write_text("Jannik Sinner,Carlos Alcaraz,Hard\n", encoding="utf-8")
    output = tmp_path / "later_out.csv"
    args = ["batch", str(batch), "--date", "2030-01-01", "--models-dir", str(trained_models_dir)]
    code, out, _ = run(capsys, *args)
    assert code == 0 and "Warning: Jannik Sinner has not played since" in out
    code, _, _ = run(capsys, *args, "--output", str(output))
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert code == 0 and "has not played since" in rows[0]["warnings"]


def test_batch_accepts_files_saved_by_excel(trained_models_dir, tmp_path, capsys):
    batch = tmp_path / "excel.csv"
    batch.write_bytes("﻿player_a;player_b;surface\r\nJannik Sinner;Carlos Alcaraz;Hard\r\n".encode())
    code, out, _ = run(capsys, "batch", str(batch), "--json", "--models-dir", str(trained_models_dir))
    results = json.loads(out)
    assert code == 0 and len(results) == 1 and results[0]["predicted_winner"]


def test_file_errors_are_clean_messages(trained_models_dir, tmp_path, capsys):
    models = ["--models-dir", str(trained_models_dir)]
    code, _, err = run(capsys, "batch", str(tmp_path / "missing.csv"), *models)
    assert code == 1 and "No such file or directory" in err and "Unexpected" not in err
    binary = tmp_path / "binary.csv"
    binary.write_bytes(b"\xff\xfe\x00bad")
    code, _, err = run(capsys, "batch", str(binary), *models)
    assert code == 1 and "not a UTF-8 text file" in err
    huge_field = tmp_path / "huge.csv"
    huge_field.write_text('"' + "x" * 200_000 + '",b,Hard\n', encoding="utf-8")
    code, _, err = run(capsys, "batch", str(huge_field), *models)
    assert code == 1 and "not a valid CSV file" in err
    good = tmp_path / "good.csv"
    good.write_text("Jannik Sinner,Carlos Alcaraz,Hard\n", encoding="utf-8")
    code, _, err = run(capsys, "batch", str(good), "--output", str(tmp_path / "no" / "out.csv"), *models)
    assert code == 1 and "No such file or directory" in err


def test_list_json_has_typed_values(trained_models_dir, capsys):
    code, out, _ = run(capsys, "list", "--json", "--models-dir", str(trained_models_dir))
    players = json.loads(out)
    assert code == 0 and players
    assert all(isinstance(p["elo"], float) and isinstance(p["wins"], int) for p in players)
    assert all(p["atp_rank"] is None or isinstance(p["atp_rank"], int) for p in players)


def test_batch_limits(trained_models_dir, tmp_path, capsys, monkeypatch):
    batch = tmp_path / "many.csv"
    batch.write_text("a,b,Hard\n" * 5)
    code, _, err = run(capsys, "batch", str(batch), "--max-rows", "2", "--models-dir", str(trained_models_dir))
    assert code == 1 and "more than 2 rows" in err
    batch.write_text("only,two\n")
    code, _, err = run(capsys, "batch", str(batch), "--models-dir", str(trained_models_dir))
    assert code == 1 and "expected" in err
    monkeypatch.setattr(cli, "MAX_BATCH_BYTES", 20)
    batch.write_text("Jannik Sinner,Carlos Alcaraz,Hard\n")
    code, _, err = run(capsys, "batch", str(batch), "--models-dir", str(trained_models_dir))
    assert code == 1 and "larger than" in err


def test_batch_rows_with_an_invalid_best_of_are_errors(trained_models_dir, tmp_path, capsys):
    batch = tmp_path / "best_of.csv"
    batch.write_text("Jannik Sinner,Carlos Alcaraz,Hard,4\nJannik Sinner,Carlos Alcaraz,Hard,3\n", encoding="utf-8")
    code, out, _ = run(capsys, "batch", str(batch), "--json", "--models-dir", str(trained_models_dir))
    results = json.loads(out)
    assert code == 1
    assert results[0]["error"] == "best_of must be 3 or 5." and results[1]["predicted_winner"]


def test_control_characters_never_reach_the_terminal(trained_models_dir, tmp_path, capsys):
    batch = tmp_path / "escape.csv"
    hostile = "Evil\x1b]0;owned\x07 Name"
    batch.write_text(f"{hostile},Carlos Alcaraz,Hard\nJannik Sinner,Carlos Alcaraz,Hard\n", encoding="utf-8")
    code, out, err = run(capsys, "batch", str(batch), "--models-dir", str(trained_models_dir))
    assert code == 1 and "Evil?]0;owned? Name" in out
    assert "\x1b" not in out + err and "\x07" not in out + err
    code, out, err = run(capsys, "predict", "Sin\x1b[2Jner", "Alcaraz", "Hard", "--models-dir", str(trained_models_dir))
    assert "\x1b" not in out + err


def test_log_level_from_the_environment_is_validated(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ATP_LOG_LEVEL", "verbose")
    with pytest.raises(SystemExit) as excinfo:
        main(["runs", "--models-dir", str(tmp_path)])
    assert excinfo.value.code == 2 and "ATP_LOG_LEVEL must be one of" in capsys.readouterr().err
    monkeypatch.setenv("ATP_LOG_LEVEL", " ")
    assert main(["runs", "--models-dir", str(tmp_path)]) == 0


@pytest.mark.parametrize(
    "args",
    [
        ["list"],
        ["list", "--sort", "wins", "--surface", "Clay", "--json"],
        ["player", "Jannik Sinner"],
        ["player", "sinner", "--json"],
        ["h2h", "Sinner", "Alcaraz"],
        ["h2h", "Sinner", "Alcaraz", "--json"],
        ["importance", "--top", "5"],
        ["importance", "--json"],
        ["shap", "--max-samples", "200"],
        ["shap", "--json", "--threshold", "1"],
        ["runs"],
        ["runs", "--json"],
    ],
)
def test_read_only_commands(trained_models_dir, capsys, args):
    code, out, _ = run(capsys, *args, "--models-dir", str(trained_models_dir))
    assert code == 0 and out.strip()
    if "--json" in args:
        json.loads(out)


def test_models_dir_from_environment(trained_models_dir, monkeypatch, capsys):
    monkeypatch.setenv("ATP_MODELS_DIR", str(trained_models_dir))
    code, out, _ = run(capsys, "runs")
    assert code == 0 and "Run" in out


def test_empty_environment_variable_means_the_default(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ATP_MODELS_DIR", " ")
    code, out, _ = run(capsys, "runs")
    assert code == 0 and "No runs in models" in out


def test_runs_on_empty_directory(tmp_path, capsys):
    code, out, _ = run(capsys, "runs", "--models-dir", str(tmp_path))
    assert code == 0 and "No runs" in out


def test_train_with_options_and_log_file(data_dir, tmp_path, capsys):
    log = tmp_path / "train.log"
    code = main(
        [
            "--log-file",
            str(log),
            "train",
            "--data-dir",
            str(data_dir),
            "--models-dir",
            str(tmp_path / "m"),
            "--max-rounds",
            "30",
            "--from-year",
            "2015",
            "--to-year",
            "2019",
            "--calibration",
            "isotonic",
            "--no-refit",
            "--per-surface",
            "--elo-k",
            "24",
            "--time-decay",
            "0.9",
            "--threads",
            "2",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0 and "Isotonic calibration" in out and "Saved run" in out
    assert "Saved model: the evaluated one" in out
    assert "Built" in log.read_text()


def test_train_prints_cross_validation_and_surface_choices(data_dir, tmp_path, capsys, monkeypatch):
    from atp_core import model

    monkeypatch.setattr(model, "MIN_MATCHES_PER_SURFACE_MODEL", 50)
    args = ["train", "--data-dir", str(data_dir), "--models-dir", str(tmp_path / "m"), "--max-rounds", "30"]
    code, out, _ = run(capsys, *args, "--cv-folds", "2", "--per-surface")
    assert code == 0
    assert "Cross-validation (2 expanding-window folds): accuracy" in out
    assert "Per-surface models (chosen on the validation period):" in out
    assert "Saved model: retrained with the same settings" in out


def test_predict_table_shows_warnings(trained_models_dir, capsys):
    args = ["predict", "Jannik Sinner", "Carlos Alcaraz", "Hard", "--date", "2030-01-01"]
    code, out, _ = run(capsys, *args, "--models-dir", str(trained_models_dir))
    assert code == 0 and "Warning: Jannik Sinner has not played since" in out


def test_runs_lists_unreadable_runs(trained_models_dir, tmp_path, capsys):
    import shutil

    models = tmp_path / "models"
    shutil.copytree(trained_models_dir, models)
    (models / "20000101T000000Z").mkdir()
    (models / "20000101T000000Z" / "manifest.json").write_text("{broken", encoding="utf-8")
    code, out, _ = run(capsys, "runs", "--models-dir", str(models))
    assert code == 0 and "20000101T000000Z  unreadable" in out


def test_debug_shows_tracebacks_for_file_errors(trained_models_dir, tmp_path):
    args = ["batch", str(tmp_path / "missing.csv"), "--models-dir", str(trained_models_dir)]
    with pytest.raises(OSError):
        main(["--log-level", "ERROR", "--debug", *args])


def test_printable_writer_replaces_control_characters():
    import io

    writer = cli._PrintableWriter(io.StringIO())
    writer.write("a\x1b[31mb\tc\n")
    assert writer.getvalue() == "a?[31mb\tc\n"


def test_train_refits_on_every_period_by_default(trained_models_dir, capsys):
    code, out, _ = run(capsys, "evaluate", "--models-dir", str(trained_models_dir))
    assert code == 0 and "Saved model: retrained with the same settings on all" in out
    code, out, _ = run(capsys, "evaluate", "--models-dir", str(trained_models_dir), "--json")
    refit = json.loads(out)["refit"]
    assert refit["models"]["all"]["matches"] > 0 and refit["to"]


def test_calibration_without_no_refit_is_a_clean_error(data_dir, tmp_path, capsys):
    code, _, err = run(
        capsys, "train", "--data-dir", str(data_dir), "--models-dir", str(tmp_path), "--calibration", "isotonic"
    )
    assert code == 1 and "--no-refit" in err


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores directory permissions")
def test_train_checks_the_models_directory_before_training(data_dir, tmp_path, capsys):
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o500)
    code, _, err = run(capsys, "train", "--data-dir", str(data_dir), "--models-dir", str(locked / "models"))
    assert code == 1 and "Permission denied" in err and "Built" not in err
    code, _, err = run(capsys, "train", "--data-dir", str(data_dir), "--models-dir", str(locked))
    assert code == 1 and "Cannot write to" in err
    locked.chmod(0o700)


def test_train_without_data_is_a_clean_error(tmp_path, capsys):
    code, _, err = run(capsys, "train", "--data-dir", str(tmp_path / "none"), "--models-dir", str(tmp_path))
    assert code == 1 and "atp download" in err


def test_unexpected_errors(monkeypatch, tmp_path, capsys):
    def boom(args):
        raise RuntimeError("secret detail")

    monkeypatch.setattr(cli, "cmd_runs", boom)
    code, _, err = run(capsys, "runs", "--models-dir", str(tmp_path))
    assert code == 3 and "RuntimeError" in err and "secret detail" not in err
    with pytest.raises(RuntimeError):
        main(["--debug", "runs", "--models-dir", str(tmp_path)])

    def interrupted(args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "cmd_runs", interrupted)
    assert main(["runs", "--models-dir", str(tmp_path)]) == 130


def test_unwritable_log_file_is_a_usage_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--log-file", str(tmp_path / "missing" / "x.log"), "runs", "--models-dir", str(tmp_path)])
    assert excinfo.value.code == 2
    assert "cannot open the log file" in capsys.readouterr().err


def test_debug_logs_traceback_for_expected_errors(tmp_path, capsys):
    assert main(["--debug", "evaluate", "--models-dir", str(tmp_path)]) == 1
    assert "Traceback" in capsys.readouterr().err


def test_module_entry_point():
    result = subprocess.run([sys.executable, "-m", "atp_core.cli", "--version"], capture_output=True, text=True)
    assert result.returncode == 0 and "atp 1." in result.stdout
