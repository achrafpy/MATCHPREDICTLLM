"""Synthetic verification only; no invented fixture prediction is published."""
import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

from matchpredict.__main__ import download, download_openfootball
from matchpredict.model import Match, Model, Players, backtest, load_matches, timestamp, two_months_before


AS_OF = datetime(2026, 10, 25, 18, tzinfo=timezone.utc)


def fixtures(dribbles=True):
    rng = np.random.default_rng(12)
    result = []
    # 96 synthetic fixtures, four simultaneous games in each of 24 rounds.
    for round_id in range(24):
        order = rng.permutation(8)
        at = AS_OF - timedelta(days=60-round_id*2)
        for i in range(0, 8, 2):
            home, away = f"T{order[i]}", f"T{order[i+1]}"
            result.append(Match(str(len(result)), at, at+timedelta(hours=2),
                                "LaLiga" if round_id % 3 else "UCL", home, away,
                                int(rng.poisson(1.5)), int(rng.poisson(1.1)),
                                float(rng.integers(3, 18)) if dribbles else None,
                                float(rng.integers(3, 18)) if dribbles else None))
    return result


def write_csv(path, fields, values):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(values)


def export_matches(path, matches):
    fields = ["match_id", "kickoff", "completed_at", "competition", "home_team", "away_team",
              "home_goals", "away_goals", "home_dribbles", "away_dribbles"]
    write_csv(path, fields, [dict(zip(fields, [m.id, m.kickoff.isoformat(), m.completed.isoformat(),
                                            m.competition, m.home, m.away, m.hg, m.ag,
                                            m.hd if m.hd is not None else "",
                                            m.ad if m.ad is not None else ""])) for m in matches])


class ModelTests(unittest.TestCase):
    def test_probability_grid_conserves_mass(self):
        report = Model(fixtures(), AS_OF).predict("T0", "T1")
        self.assertAlmostEqual(sum(report["probabilities"].values()), 1, places=9)
        self.assertLess(report["omitted_probability_mass"], 1e-9)
        self.assertTrue(report["features"]["team_dribbles"])
        self.assertGreater(report["top_scores"][0]["probability"], 0)

    def test_future_and_in_progress_results_cannot_change_forecast(self):
        matches = fixtures()
        before = Model(matches, AS_OF).predict("T0", "T1")
        matches.extend([
            Match("future", AS_OF+timedelta(days=1), AS_OF+timedelta(days=2), "UCL", "T0", "T1", 99, 0, 1000, 0),
            Match("live", AS_OF-timedelta(minutes=30), AS_OF+timedelta(hours=2), "UCL", "T0", "T1", 99, 0, 1000, 0),
        ])
        self.assertEqual(before, Model(matches, AS_OF).predict("T0", "T1"))

    def test_features_do_not_read_later_results(self):
        model = Model(fixtures(), AS_OF)
        early = model.matches[4]
        a = model._row(early.id, early.home, early.away, True, early.kickoff)
        model.matches = [m for m in model.matches if m.completed < early.kickoff]
        b = model._row(early.id, early.home, early.away, True, early.kickoff)
        self.assertEqual(a, b)

    def test_walk_forward_runs_and_reports_baseline(self):
        result = backtest(fixtures(False), AS_OF, limit=5)
        self.assertEqual(result["evaluated"], 5)
        self.assertEqual(result["skipped"], [])
        self.assertTrue(np.isfinite(result["baseline_mean_score_negative_log_likelihood"]))
        self.assertGreaterEqual(result["three_way_brier_score"], 0)

    def test_missing_and_partial_data_are_not_silently_filled(self):
        with self.assertRaisesRegex(ValueError, "at least 40"):
            Model(fixtures()[:5], AS_OF)
        mixed = fixtures()
        m = mixed[0]
        mixed[0] = Match(m.id, m.kickoff, m.completed, m.competition, m.home, m.away, m.hg, m.ag)
        with self.assertRaisesRegex(ValueError, "coverage"):
            Model(mixed, AS_OF)
        result = Model(fixtures(False), AS_OF).predict("T0", "T1")
        self.assertFalse(result["features"]["team_dribbles"])

    def test_calendar_window_and_timezone(self):
        self.assertEqual(two_months_before(timestamp("2026-10-31T20:00:00Z")).isoformat(), "2026-08-31T20:00:00+00:00")
        self.assertEqual(two_months_before(timestamp("2026-04-30T20:00:00Z")).day, 28)
        with self.assertRaisesRegex(ValueError, "timezone"):
            timestamp("2026-10-25T21:00:00")

    def test_stale_target_form_blocks_prediction(self):
        with self.assertRaisesRegex(ValueError, "Stale data"):
            Model(fixtures(), AS_OF+timedelta(days=3)).predict("T0", "T1")

    def test_transfer_uses_stable_player_id_and_current_lineup(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats, lineups = Path(tmp)/"players.csv", Path(tmp)/"lineups.csv"
            at = AS_OF-timedelta(days=10)
            matches = [Match("old", at, at+timedelta(hours=2), "LaLiga", "OldClub", "Other", 2, 0)]
            write_csv(stats, ["match_id", "team", "player_id", "minutes", "goals", "dribbles"], [
                {"match_id": "old", "team": "OldClub", "player_id": "transferred", "minutes": 90, "goals": 2, "dribbles": 8},
                {"match_id": "old", "team": "Other", "player_id": "control", "minutes": 90, "goals": 0, "dribbles": 0}])
            lineup = [{"match_id": "target", "team": "NewClub", "player_id": "transferred" if i == 0 else f"new{i}",
                       "expected_minutes": 90, "known_at": (AS_OF-timedelta(hours=1)).isoformat()} for i in range(11)]
            write_csv(lineups, list(lineup[0]), lineup)
            players = Players(str(stats), str(lineups), matches)
            with_history = players.features("target", "NewClub", AS_OF, two_months_before(AS_OF))
            self.assertGreater(with_history[0], 0)
            # Move the cutoff before his old-club performance: it must disappear.
            self.assertEqual(players.features("target", "NewClub", AS_OF, AS_OF-timedelta(days=1)), [0, 0])
            lineup[0]["known_at"] = (AS_OF+timedelta(hours=1)).isoformat()
            write_csv(lineups, list(lineup[0]), lineup)
            with self.assertRaisesRegex(ValueError, "pre-cutoff"):
                Players(str(stats), str(lineups), matches).features("target", "NewClub", AS_OF, two_months_before(AS_OF))

    def test_full_player_pipeline_fits_and_predicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            matches = fixtures()
            stats, lineups = [], []
            for m in matches:
                for team in (m.home, m.away):
                    for i in range(11):
                        pid = f"{team}-{i}"
                        stats.append(dict(match_id=m.id, team=team, player_id=pid, minutes=90,
                                          goals=int(i == 0)*(m.hg if team == m.home else m.ag), dribbles=i % 3))
                        lineups.append(dict(match_id=m.id, team=team, player_id=pid, expected_minutes=90,
                                            known_at=(m.kickoff-timedelta(hours=1)).isoformat()))
            for team in ("T0", "T1"):
                for i in range(11):
                    lineups.append(dict(match_id="target", team=team, player_id=f"{team}-{i}",
                                        expected_minutes=90, known_at=(AS_OF-timedelta(hours=1)).isoformat()))
            sp, lp = Path(tmp)/"stats.csv", Path(tmp)/"lineups.csv"
            write_csv(sp, list(stats[0]), stats); write_csv(lp, list(lineups[0]), lineups)
            players = Players(str(sp), str(lp), matches)
            result = Model(matches, AS_OF, players).predict("T0", "T1")
            self.assertTrue(result["features"]["player_form_and_lineups"])

    def test_csv_duplicate_and_negative_goals_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"m.csv"
            export_matches(path, fixtures()[:2])
            self.assertEqual(len(load_matches(path)), 2)
            export_matches(path, fixtures()[:1]*2)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_matches(path)
            m = fixtures()[0]
            export_matches(path, [Match(m.id, m.kickoff, m.completed, m.competition, m.home, m.away, -1, 0)])
            with self.assertRaisesRegex(ValueError, "goals"):
                load_matches(path)

    def test_command_line_predicts_from_synthetic_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"m.csv"; export_matches(path, fixtures())
            process = subprocess.run([sys.executable, "-m", "matchpredict", "predict", "--matches", str(path),
                                      "--home", "T0", "--away", "T1", "--as-of", AS_OF.isoformat(),
                                      "--kickoff", "2026-10-25T21:00:00+01:00"], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(json.loads(process.stdout)["training_matches"], 96)

    def test_free_csv_import_keeps_provenance_and_delays_results(self):
        content = b"Date,HomeTeam,AwayTeam,FTHG,FTAG\n20/10/2026,Barcelona,Real Madrid,1,1\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"free.csv"
            with patch("matchpredict.__main__.urlopen", return_value=io.BytesIO(content)):
                result = download("2627", "SP1", path)
            self.assertEqual(result["downloaded_matches"], 1)
            self.assertEqual(load_matches(path)[0].completed, timestamp("2026-10-22T00:00:00Z"))
            with patch("matchpredict.__main__.urlopen", return_value=io.BytesIO(content)):
                with self.assertRaisesRegex(ValueError, "exists"):
                    download("2627", "SP1", path)

    def test_community_import_preserves_source_and_ignores_unplayed_games(self):
        content = json.dumps({"matches": [
            {"date": "2026-10-01", "team1": "A", "team2": "B", "score": {"ft": [2, 1]}},
            {"date": "2026-10-02", "team1": "C", "team2": "D", "score": [0, 0]},
            {"date": "2026-10-25", "team1": "A", "team2": "B"}]}).encode()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"community.csv"
            with patch("matchpredict.__main__.urlopen", return_value=io.BytesIO(content)):
                result = download_openfootball("2627", "SP1", path)
            self.assertEqual(result["downloaded_matches"], 1)
            self.assertEqual(result["ambiguous_score_entries_excluded"], 1)
            self.assertEqual(result["latest_match_date"], "2026-10-01")
            self.assertEqual(len(result["source_sha256"]), 64)
            self.assertEqual(load_matches(path)[0].hg, 2)


if __name__ == "__main__":
    unittest.main()
