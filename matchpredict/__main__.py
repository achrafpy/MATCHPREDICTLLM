"""Command line entry point; no paid APIs or secrets required."""
import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from .model import Model, Players, backtest, load_matches, timestamp


def download_openfootball(season, league, output):
    if len(season) != 4 or not season.isdigit() or league != "SP1":
        raise ValueError("OpenFootball importer currently supports SP1 and four-digit seasons only")
    year = 2000+int(season[:2])
    if int(season[2:]) != (year+1) % 100:
        raise ValueError("Season years must be consecutive")
    url = f"https://raw.githubusercontent.com/openfootball/football.json/master/{year}-{season[2:]}/es.1.json"
    with urlopen(Request(url, headers={"User-Agent": "MatchPredict/0.1"}), timeout=30) as response:
        content = response.read(10_000_001)
    if len(content) > 10_000_000:
        raise ValueError("Unexpectedly large JSON response")
    payload = json.loads(content)
    imported, ambiguous = [], 0
    digest = hashlib.sha256(content).hexdigest()
    fetched_at = datetime.now(timezone.utc).isoformat()
    for match in payload["matches"]:
        score_field = match.get("score")
        if score_field is None:
            continue
        if not isinstance(score_field, dict):
            # Bare arrays do not explicitly identify a full-time result.
            ambiguous += 1
            continue
        score = score_field.get("ft")
        if score is None:
            continue
        if len(score) != 2 or any(type(x) is not int or x < 0 for x in score):
            raise ValueError("Invalid final score in community feed")
        day = datetime.strptime(match["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        imported.append({"match_id": f"SP1:{match['date']}:{match['team1']}:{match['team2']}",
                         "kickoff": day.isoformat(), "completed_at": (day+timedelta(days=2)).isoformat(),
                         "competition": "LaLiga", "home_team": match["team1"], "away_team": match["team2"],
                         "home_goals": score[0], "away_goals": score[1], "source_url": url,
                         "source_sha256": digest, "retrieved_at": fetched_at,
                         "time_precision": "date_only_conservative"})
    if not imported:
        raise ValueError("Community feed contains no completed matches")
    output = Path(output)
    if output.exists():
        raise ValueError(f"Output exists; choose another path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(imported[0]))
        writer.writeheader(); writer.writerows(imported)
    return {"downloaded_matches": len(imported), "output": str(output), "source": url,
            "ambiguous_score_entries_excluded": ambiguous,
            "source_sha256": digest, "latest_match_date": max(r["kickoff"][:10] for r in imported),
            "limitations": ["Community-maintained results; not independently verified.",
                            "No dribbles, players or Champions League; freshness must be checked.",
                            "Date-only timestamps; results delayed up to 48 hours."]}


def download(season, league, output):
    # Restrict user inputs so the URL always addresses the documented free feed.
    if len(season) != 4 or not season.isdigit() or league not in {"SP1", "E0", "D1", "I1", "F1", "N1", "P1", "B1"}:
        raise ValueError("Use a four-digit season (2627) and a supported league code")
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
    request = Request(url, headers={"User-Agent": "MatchPredict/0.1 (research; free CSV)"})
    with urlopen(request, timeout=30) as response:
        content = response.read(10_000_001)
    if len(content) > 10_000_000:
        raise ValueError("Unexpectedly large CSV response")
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
    if not {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"} <= set(reader.fieldnames or []):
        raise ValueError("Response is not the expected football-data results CSV")
    converted = []
    for r in reader:
        if r["FTHG"] == "" or r["FTAG"] == "":
            continue
        date = None
        for fmt in ("%d/%m/%Y", "%d/%m/%y"):
            try:
                date = datetime.strptime(r["Date"], fmt).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                pass
        if date is None:
            raise ValueError(f"Invalid source date: {r['Date']}")
        # The feed's time zone is not assumed. Date-only records are made usable
        # only after TWO UTC midnights, avoiding same-day result look-ahead.
        converted.append({"match_id": f"{league}:{r['Date']}:{r['HomeTeam']}:{r['AwayTeam']}",
                          "kickoff": date.isoformat(),
                          "completed_at": (date+timedelta(days=2)).isoformat(),
                          "competition": "LaLiga" if league == "SP1" else league,
                          "home_team": r["HomeTeam"], "away_team": r["AwayTeam"],
                          "home_goals": int(r["FTHG"]), "away_goals": int(r["FTAG"]),
                          "source_url": url, "time_precision": "date_only_conservative"})
    if not converted:
        raise ValueError("Feed contains no completed matches; output was not written")
    output = Path(output)
    if output.exists():
        raise ValueError(f"Output exists; choose another path to preserve it: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(converted[0]))
        writer.writeheader(); writer.writerows(converted)
    return {"downloaded_matches": len(converted), "output": str(output), "source": url,
            "limitations": ["Results only: no Champions League, dribbles or player data.",
                            "Date-only timestamps; newest results delayed by up to 48 hours."]}


def main():
    parser = argparse.ArgumentParser(description="Football score probabilities, not guaranteed results")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download", help="Import free community or football-data results")
    fetch.add_argument("--source", choices=["openfootball", "football-data"], default="openfootball")
    fetch.add_argument("--season", default="2627")
    fetch.add_argument("--league", default="SP1")
    fetch.add_argument("--output", required=True)
    for name in ("predict", "backtest"):
        p = commands.add_parser(name)
        p.add_argument("--matches", required=True)
        p.add_argument("--as-of", required=True, help="Aware ISO timestamp: only earlier results may be used")
        p.add_argument("--players")
        p.add_argument("--lineups")
        if name == "predict":
            p.add_argument("--home", required=True)
            p.add_argument("--away", required=True)
            p.add_argument("--kickoff", required=True, help="User-supplied fixture time; verify independently")
            p.add_argument("--match-id", default="target")
    args = parser.parse_args()
    try:
        if args.command == "download":
            importer = download_openfootball if args.source == "openfootball" else download
            result = importer(args.season, args.league, args.output)
        else:
            if bool(args.players) != bool(args.lineups):
                raise ValueError("Supply --players and --lineups together")
            matches = load_matches(args.matches)
            as_of = timestamp(args.as_of)
            if as_of > datetime.now(timezone.utc):
                raise ValueError("--as-of is in the future; use the actual prediction cutoff")
            players = Players(args.players, args.lineups, matches) if args.players else None
            if args.command == "predict":
                kickoff = timestamp(args.kickoff)
                if as_of > kickoff:
                    raise ValueError("A pre-match forecast cutoff cannot be after kickoff")
                result = Model(matches, as_of, players).predict(args.home, args.away, args.match_id)
                result["fixture_kickoff_unverified"] = kickoff.isoformat()
                result["data_file"] = str(Path(args.matches).resolve())
            else:
                result = backtest(matches, as_of, players)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, KeyError, OSError, URLError) as exc:
        parser.exit(2, f"No se pudo completar: {exc}\n")


if __name__ == "__main__":
    main()
