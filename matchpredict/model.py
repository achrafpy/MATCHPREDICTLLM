"""Regularised Poisson attack/defence model with strictly historical features."""
from __future__ import annotations

import calendar
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


def timestamp(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"Timestamp requires a timezone: {value}")
    return dt.astimezone(timezone.utc)


def two_months_before(dt: datetime) -> datetime:
    month_index = dt.year * 12 + dt.month - 1 - 2
    year, month = divmod(month_index, 12)
    month += 1
    return dt.replace(year=year, month=month,
                      day=min(dt.day, calendar.monthrange(year, month)[1]))


def rows(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def nonnegative(value: str, name: str, integer: bool = False) -> float:
    n = float(value)
    if not np.isfinite(n) or n < 0 or (integer and n != int(n)):
        raise ValueError(f"Invalid {name}: {value}")
    return n


@dataclass(frozen=True)
class Match:
    id: str
    kickoff: datetime
    completed: datetime
    competition: str
    home: str
    away: str
    hg: int
    ag: int
    hd: float | None = None
    ad: float | None = None


def load_matches(path: str | Path) -> list[Match]:
    result, seen = [], set()
    for r in rows(path):
        mid = r["match_id"]
        if not mid or mid in seen:
            raise ValueError(f"Empty/duplicate match_id: {mid}")
        seen.add(mid)
        kickoff, completed = timestamp(r["kickoff"]), timestamp(r["completed_at"])
        if completed <= kickoff or r["home_team"] == r["away_team"]:
            raise ValueError(f"Invalid fixture: {mid}")
        dribbles = [r.get(k, "") for k in ("home_dribbles", "away_dribbles")]
        if bool(dribbles[0]) != bool(dribbles[1]):
            raise ValueError(f"Both dribble fields are required together: {mid}")
        result.append(Match(mid, kickoff, completed, r["competition"],
                            r["home_team"], r["away_team"],
                            int(nonnegative(r["home_goals"], "goals", True)),
                            int(nonnegative(r["away_goals"], "goals", True)),
                            *(nonnegative(v, "dribbles") if v else None for v in dribbles)))
    return sorted(result, key=lambda m: (m.kickoff, m.id))


class Players:
    """Stable IDs carry a player's past form across clubs; lineups assign their club.

    Historical lineup snapshots MUST be pre-match estimates, never actual minutes.
    Player match statistics become usable only after their match completed.
    """

    def __init__(self, stats_path: str, lineups_path: str, matches: list[Match]):
        self.stats = rows(stats_path)
        self.lineups = rows(lineups_path)
        self.matches = {m.id: m for m in matches}
        seen = set()
        for r in self.stats:
            key = r["match_id"], r["player_id"]
            if key in seen or r["match_id"] not in self.matches:
                raise ValueError(f"Duplicate player or unknown match: {key}")
            seen.add(key)
            m = self.matches[r["match_id"]]
            if r["team"] not in (m.home, m.away):
                raise ValueError(f"Player team does not belong to match: {key}")
            for name in ("minutes", "goals", "dribbles"):
                r[name] = nonnegative(r[name], name)
            if r["minutes"] > 130 or (r["minutes"] == 0 and (r["goals"] or r["dribbles"])):
                raise ValueError(f"Invalid player minutes: {key}")
        seen.clear()
        for r in self.lineups:
            key = r["match_id"], r["team"], r["player_id"]
            if key in seen:
                raise ValueError(f"Duplicate lineup player: {key}")
            seen.add(key)
            r["known_at"] = timestamp(r["known_at"])
            r["expected_minutes"] = nonnegative(r["expected_minutes"], "expected_minutes")
            if r["expected_minutes"] > 90:
                raise ValueError("Expected minutes cannot exceed 90")
            if r["match_id"] in self.matches:
                m = self.matches[r["match_id"]]
                if r["team"] not in (m.home, m.away) or r["known_at"] >= m.kickoff:
                    raise ValueError(f"Invalid historical lineup snapshot: {key}")

    def features(self, match_id, team, at, since):
        lineup = [r for r in self.lineups if r["match_id"] == match_id and r["team"] == team]
        if not lineup or any(r["known_at"] >= at for r in lineup):
            raise ValueError(f"Missing pre-cutoff lineup: {match_id}/{team}")
        if abs(sum(r["expected_minutes"] for r in lineup) - 990) > 0.01:
            raise ValueError(f"Expected minutes must total 990: {match_id}/{team}")
        past = [r for r in self.stats
                if since <= self.matches[r["match_id"]].kickoff
                and self.matches[r["match_id"]].completed < at]
        if not past:
            return [0.0, 0.0]
        total_minutes = sum(r["minutes"] for r in past)
        prior = np.array([sum(r[k] for r in past) for k in ("goals", "dribbles")]) / max(total_minutes, 1)
        projected = np.zeros(2)
        for player in lineup:
            history = [r for r in past if r["player_id"] == player["player_id"]]
            mins = sum(r["minutes"] for r in history)
            counts = np.array([sum(r[k] for r in history) for k in ("goals", "dribbles")])
            # 180 pseudo-minutes shrink noisy two-month player samples.
            projected += (counts + 180 * prior) / (mins + 180) * player["expected_minutes"]
        return np.log1p(projected).tolist()


def form(matches, team, at, dribbles):
    past = [m for m in matches if m.completed < at and team in (m.home, m.away)]
    past = sorted(past, key=lambda m: m.completed)[-5:]
    if not past:
        return [1.35, 1.35] + ([0.0] if dribbles else [])
    scored = [m.hg if m.home == team else m.ag for m in past]
    conceded = [m.ag if m.home == team else m.hg for m in past]
    result = [float(np.mean(scored)), float(np.mean(conceded))]
    if dribbles:
        result.append(float(np.log1p(np.mean([m.hd if m.home == team else m.ad for m in past]))))
    return result


class Model:
    def __init__(self, matches: list[Match], as_of: datetime, players: Players | None = None,
                 ridge: float = 8.0, half_life: float = 30.0, min_matches: int = 40):
        if ridge <= 0 or half_life <= 0:
            raise ValueError("ridge and half_life must be positive")
        self.as_of, self.since, self.players = as_of, two_months_before(as_of), players
        self.matches = [m for m in matches if self.since <= m.kickoff and m.completed < as_of]
        if len(self.matches) < min_matches:
            raise ValueError(f"Need at least {min_matches} completed matches in the two-month window; got {len(self.matches)}")
        self.teams = sorted({t for m in self.matches for t in (m.home, m.away)})
        self.indices = {t: i for i, t in enumerate(self.teams)}
        flags = [m.hd is not None and m.ad is not None for m in self.matches]
        if any(flags) and not all(flags):
            raise ValueError("Dribble coverage must be complete or entirely omitted; partial coverage is unsafe")
        self.dribbles = all(flags)
        raw, goals, weights = [], [], []
        for m in self.matches:
            for team, other, home, y in [(m.home, m.away, True, m.hg), (m.away, m.home, False, m.ag)]:
                raw.append(self._row(m.id, team, other, home, m.kickoff))
                goals.append(y)
                weights.append(2 ** (-(as_of - m.completed).total_seconds() / 86400 / half_life))
        x = np.array(raw, dtype=float)
        self.base_columns = 2 + 2 * len(self.teams)
        features = x[:, self.base_columns:]
        self.mean = features.mean(axis=0)
        self.scale = features.std(axis=0)
        self.scale[self.scale < 1e-8] = 1
        x[:, self.base_columns:] = (features - self.mean) / self.scale
        y, w = np.array(goals), np.array(weights)
        penalty = np.full(x.shape[1], ridge)
        penalty[:2] = 0.1

        def objective(b):
            eta = x @ b
            mu = np.exp(eta)
            return float(np.sum(w * (mu - y * eta)) + 0.5 * np.dot(penalty, b*b)), x.T @ (w * (mu-y)) + penalty*b

        fit = minimize(objective, np.zeros(x.shape[1]), method="L-BFGS-B", jac=True,
                       bounds=[(-5, 5)] * x.shape[1], options={"maxiter": 2000, "ftol": 1e-11})
        if not fit.success or not np.all(np.isfinite(fit.x)):
            raise ValueError(f"Poisson optimisation failed: {fit.message}")
        self.coef = fit.x
        self.ridge, self.half_life = ridge, half_life

    def _row(self, match_id, team, opponent, home, at):
        n = len(self.teams)
        row = np.zeros(2 + 2*n)
        row[0], row[1] = 1, int(home)
        row[2+self.indices[team]] = 1
        row[2+n+self.indices[opponent]] = 1
        own, opp = form(self.matches, team, at, self.dribbles), form(self.matches, opponent, at, self.dribbles)
        result = row.tolist() + own + opp
        if self.players:
            result += self.players.features(match_id, team, at, self.since)
            result += self.players.features(match_id, opponent, at, self.since)
        return result

    def predict(self, home, away, match_id="target", top=5):
        if home == away:
            raise ValueError("Teams must differ")
        for team in (home, away):
            if team not in self.indices:
                raise ValueError(f"Unknown team: {team}; use the exact CSV team name")
            if sum(team in (m.home, m.away) for m in self.matches) < 3:
                raise ValueError(f"Need at least three completed matches for {team}")
            latest = max(m.completed for m in self.matches if team in (m.home, m.away))
            if (self.as_of-latest).total_seconds() > 14*86400:
                raise ValueError(f"Stale data for {team}: latest available result {latest.isoformat()}; "
                                 "need a result within 14 days of the prediction cutoff")
        # Disconnected competition pools cannot identify relative team strengths.
        reached, changed = {home}, True
        while changed:
            changed = False
            for m in self.matches:
                if (m.home in reached or m.away in reached) and not {m.home, m.away} <= reached:
                    reached.update((m.home, m.away)); changed = True
        if away not in reached:
            raise ValueError("Teams belong to disconnected opponent networks")
        x = np.array([self._row(match_id, home, away, True, self.as_of),
                      self._row(match_id, away, home, False, self.as_of)])
        x[:, self.base_columns:] = (x[:, self.base_columns:] - self.mean) / self.scale
        rates = np.exp(x @ self.coef)
        if not np.all(np.isfinite(rates)) or max(rates) > 15:
            raise ValueError("Implausible goal rates; inspect data and feature extrapolation")
        cap = max(10, int(poisson.ppf(1 - 1e-12, max(rates))))
        grid = np.outer(poisson.pmf(np.arange(cap+1), rates[0]), poisson.pmf(np.arange(cap+1), rates[1]))
        indices = np.argsort(grid.ravel())[::-1][:top]
        warnings = ["Experimental model; probabilities are not calibrated betting advice.",
                    "Only two months of data: substantial sampling and lineup uncertainty.",
                    "Independent Poisson does not explicitly model score correlation, red cards or injuries.",
                    "Fixture date, venue and lineup authenticity must be verified externally."]
        if not self.dribbles:
            warnings.append("Team dribbles were not supplied and are not included.")
        if not self.players:
            warnings.append("No player/lineup inputs: transfers and absences are not explicitly modelled.")
        if not any("champions" in m.competition.lower() or m.competition.upper() == "UCL" for m in self.matches):
            warnings.append("No Champions League data supplied.")
        return {"as_of": self.as_of.isoformat(), "window_start": self.since.isoformat(),
                "home": home, "away": away, "training_matches": len(self.matches),
                "competitions": sorted({m.competition for m in self.matches}),
                "latest_result_available_at": max(m.completed for m in self.matches).isoformat(),
                "expected_goals": {"home": float(rates[0]), "away": float(rates[1])},
                "probabilities": {"home_win": float(np.tril(grid, -1).sum()),
                                  "draw": float(np.trace(grid)), "away_win": float(np.triu(grid, 1).sum())},
                "top_scores": [{"home_goals": int(i // (cap+1)), "away_goals": int(i % (cap+1)),
                                "probability": float(grid.ravel()[i])} for i in indices],
                "omitted_probability_mass": float(1-grid.sum()),
                "features": {"team_dribbles": self.dribbles, "player_form_and_lineups": bool(self.players)},
                "parameters": {"ridge": self.ridge, "half_life_days": self.half_life},
                "warnings": warnings}


def backtest(matches, as_of, players=None, limit=20):
    """Walk forward in time; every fit sees only already completed matches."""
    selected = [m for m in matches if two_months_before(as_of) <= m.kickoff and m.completed < as_of]
    if len(selected) < 60:
        raise ValueError("Backtest needs at least 60 matches; 40 training plus 20 candidates")
    scores, baseline_scores, briers, skipped = [], [], [], []
    for m in sorted(selected, key=lambda m: (m.kickoff, m.id))[-limit:]:
        try:
            model = Model(matches, m.kickoff, players)
            result = model.predict(m.home, m.away, m.id)
        except ValueError as exc:
            skipped.append({"match_id": m.id, "reason": str(exc)})
            continue
        lh, la = result["expected_goals"].values()
        scores.append(float(-poisson.logpmf(m.hg, lh)-poisson.logpmf(m.ag, la)))
        bh = (sum(x.hg for x in model.matches)+1)/(len(model.matches)+1)
        ba = (sum(x.ag for x in model.matches)+1)/(len(model.matches)+1)
        baseline_scores.append(float(-poisson.logpmf(m.hg, bh)-poisson.logpmf(m.ag, ba)))
        p = np.array(list(result["probabilities"].values()))
        truth = np.zeros(3); truth[0 if m.hg > m.ag else 1 if m.hg == m.ag else 2] = 1
        briers.append(float(np.sum((p-truth)**2)))
    if not scores:
        raise ValueError(f"Backtest evaluated zero matches; first exclusions: {skipped[:3]}")
    return {"evaluated": len(scores), "skipped": skipped,
            "mean_score_negative_log_likelihood": float(np.mean(scores)),
            "baseline_mean_score_negative_log_likelihood": float(np.mean(baseline_scores)),
            "three_way_brier_score": float(np.mean(briers)),
            "note": "Lower is better. Small holdout; no calibration or superiority claim is established."}
