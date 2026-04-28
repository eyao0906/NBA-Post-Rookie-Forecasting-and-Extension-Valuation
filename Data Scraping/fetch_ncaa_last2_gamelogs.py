import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://www.sports-reference.com"
SEARCH_URL = BASE + "/cbb/search/search.fcgi?hint=&search={query}"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

COHORT_PATH = Path(DATA_DIR / "cohort_1999_2019.csv")
OUT_PATH = Path(DATA_DIR / "raw_game_logs_ncaa_last2.csv")
UNMATCHED_PATH = Path(DATA_DIR / "ncaa_unmatched_players.csv")
PROGRESS_PATH = Path(DATA_DIR / "ncaa_progress.csv")

OUTPUT_COLUMNS = [
    "SEASON_ID", "Player_ID", "Game_ID", "GAME_DATE", "MATCHUP", "WL", "MIN", "FGM", "FGA", "FG_PCT",
    "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "STL", "BLK",
    "TOV", "PF", "PTS", "PLUS_MINUS", "VIDEO_AVAILABLE", "PLAYER_NAME", "SEASON_STRING", "DATA_SOURCE",
]


@dataclass
class MatchResult:
    player_url: Optional[str]
    note: str


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name)).strip()


def parse_result_years(text: str) -> tuple[Optional[int], Optional[int]]:
    m = re.search(r"\((\d{4})-(\d{4})\)", text)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def choose_best_candidate(items, draft_year: int) -> Optional[str]:
    scored = []
    for it in items:
        name_block = it.select_one("div.search-item-name")
        url_block = it.select_one("div.search-item-url")
        if not name_block or not url_block:
            continue
        href = url_block.get_text(" ", strip=True)
        y0, y1 = parse_result_years(name_block.get_text(" ", strip=True))
        if not href.startswith("/cbb/players/"):
            continue

        score = 10_000
        if y1 is not None:
            score = abs(y1 - draft_year)
            if y1 == draft_year:
                score -= 5
            if y0 is not None and y0 <= draft_year <= y1:
                score -= 3
        scored.append((score, href))

    if not scored:
        return None
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def get_with_retries(session: requests.Session, url: str, timeout: int = 25, max_retries: int = 8):
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 429:
                time.sleep(min(120, 12 * attempt))
                continue
            if r.status_code >= 500:
                time.sleep(min(30, 3 * attempt))
                continue
            return r
        except Exception as e:  # noqa: BLE001
            last_exc = e
            time.sleep(min(30, 2 * attempt))
    raise RuntimeError(f"request_failed:{url}::{last_exc}")


def resolve_player_url(session: requests.Session, player_name: str, draft_year: int) -> MatchResult:
    q = requests.utils.quote(player_name)
    url = SEARCH_URL.format(query=q)

    try:
        r = get_with_retries(session, url, timeout=25, max_retries=8)
    except Exception as e:
        return MatchResult(None, f"search_error:{e}")

    final = r.url
    if "/cbb/players/" in final and final.endswith(".html"):
        return MatchResult(final, "direct")

    soup = BeautifulSoup(r.text, "html.parser")
    items = soup.select("div.search-item")
    if not items:
        return MatchResult(None, "no_search_results")

    best = choose_best_candidate(items, draft_year)
    if not best:
        return MatchResult(None, "no_candidate")

    return MatchResult(BASE + best, "search_result")


def get_last_two_season_end_years(player_page_url: str) -> list[int]:
    try:
        tables = pd.read_html(player_page_url)
    except Exception:
        return []
    if not tables:
        return []

    per_game = tables[0].copy()
    if "Season" not in per_game.columns:
        return []

    seasons = []
    for s in per_game["Season"].astype(str):
        m = re.match(r"^(\d{4})-(\d{2})$", s)
        if not m:
            continue
        end = 2000 + int(m.group(2)) if int(m.group(2)) <= 30 else 1900 + int(m.group(2))
        seasons.append(end)

    seasons = sorted(set(seasons))
    return seasons


def to_float(v):
    if pd.isna(v):
        return None
    if isinstance(v, str):
        v = v.strip()
        if not v:
            return None
        if v.startswith('.'):
            v = '0' + v
    try:
        return float(v)
    except Exception:
        return None


def to_int(v):
    f = to_float(v)
    return None if f is None else int(round(f))


def build_matchup(team: str, site: str, opp: str) -> str:
    site = "" if pd.isna(site) else str(site).strip()
    if site == "@":
        return f"{team} @ {opp}"
    return f"{team} vs. {opp}"


def parse_wl(result: str) -> Optional[str]:
    if pd.isna(result):
        return None
    result = str(result).strip()
    if result.startswith("W"):
        return "W"
    if result.startswith("L"):
        return "L"
    return None


def fetch_gamelog_table(url: str) -> pd.DataFrame:
    try:
        dfs = pd.read_html(url)
    except Exception:
        return pd.DataFrame()
    if not dfs:
        return pd.DataFrame()
    g = dfs[0].copy()
    if "Rk" in g.columns:
        g = g[g["Rk"].astype(str) != "Rk"]
    return g


def append_rows(path: Path, rows: list[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[OUTPUT_COLUMNS]

    exists = path.exists()
    df.to_csv(path, mode="a", header=not exists, index=False)


def mark_progress(person_id: int, player_name: str, draft_year: int, status: str, rows_written: int):
    row = pd.DataFrame([
        {
            "PERSON_ID": int(person_id),
            "PLAYER_NAME": player_name,
            "DRAFT_YEAR": int(draft_year),
            "status": status,
            "rows_written": int(rows_written),
            "ts": pd.Timestamp.utcnow().isoformat(),
        }
    ])
    exists = PROGRESS_PATH.exists()
    row.to_csv(PROGRESS_PATH, mode="a", header=not exists, index=False)


def load_processed_ids() -> set[int]:
    # Primary source: explicit per-player progress log.
    if PROGRESS_PATH.exists():
        p = pd.read_csv(PROGRESS_PATH)
        if not p.empty and "PERSON_ID" in p.columns:
            return set(pd.to_numeric(p["PERSON_ID"], errors="coerce").dropna().astype(int).tolist())

    # Bootstrap fallback: infer already-processed players from the existing partial output.
    if OUT_PATH.exists():
        try:
            out = pd.read_csv(OUT_PATH, usecols=["Player_ID"])
            if not out.empty and "Player_ID" in out.columns:
                return set(pd.to_numeric(out["Player_ID"], errors="coerce").dropna().astype(int).tolist())
        except Exception:
            pass

    return set()


def main():
    cohort = pd.read_csv(COHORT_PATH)[["PERSON_ID", "PLAYER_NAME", "SEASON"]].copy()
    cohort["PLAYER_NAME"] = cohort["PLAYER_NAME"].map(normalize_name)
    cohort["SEASON"] = cohort["SEASON"].astype(int)

    processed_ids = load_processed_ids()
    cohort = cohort[~cohort["PERSON_ID"].astype(int).isin(processed_ids)].reset_index(drop=True)

    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0"})

    name_cache: dict[tuple[str, int], MatchResult] = {}
    unmatched = []

    print(f"Remaining players to process: {len(cohort)}")

    for i, p in cohort.iterrows():
        player_id = int(p["PERSON_ID"])
        player_name = p["PLAYER_NAME"]
        draft_year = int(p["SEASON"])

        cache_key = (player_name.lower(), draft_year)
        match = name_cache.get(cache_key)
        if match is None:
            match = resolve_player_url(sess, player_name, draft_year)
            name_cache[cache_key] = match

        player_rows = []

        if match.player_url:
            season_end_years = [y for y in get_last_two_season_end_years(match.player_url) if y <= draft_year]
            season_end_years = sorted(season_end_years)[-2:]

            if season_end_years:
                slug = match.player_url.rstrip("/").split("/")[-1].replace(".html", "")
                for end_year in season_end_years:
                    g = fetch_gamelog_table(f"{BASE}/cbb/players/{slug}/gamelog/{end_year}")
                    if g.empty:
                        continue

                    for ridx, r in g.iterrows():
                        season_string = f"{end_year-1}-{str(end_year)[-2:]}"
                        team = str(r.get("Team", "")).strip()
                        opp = str(r.get("Opp", "")).strip()
                        site = r.get("Unnamed: 5", "")
                        player_rows.append(
                            {
                                "SEASON_ID": int(end_year),
                                "Player_ID": player_id,
                                "Game_ID": f"NCAA_{player_id}_{end_year}_{int(ridx)+1}",
                                "GAME_DATE": r.get("Date"),
                                "MATCHUP": build_matchup(team, site, opp),
                                "WL": parse_wl(r.get("Result")),
                                "MIN": to_int(r.get("MP")),
                                "FGM": to_int(r.get("FG")),
                                "FGA": to_int(r.get("FGA")),
                                "FG_PCT": to_float(r.get("FG%")),
                                "FG3M": to_int(r.get("3P")),
                                "FG3A": to_int(r.get("3PA")),
                                "FG3_PCT": to_float(r.get("3P%")),
                                "FTM": to_int(r.get("FT")),
                                "FTA": to_int(r.get("FTA")),
                                "FT_PCT": to_float(r.get("FT%")),
                                "OREB": to_int(r.get("ORB")),
                                "DREB": to_int(r.get("DRB")),
                                "REB": to_int(r.get("TRB")),
                                "AST": to_int(r.get("AST")),
                                "STL": to_int(r.get("STL")),
                                "BLK": to_int(r.get("BLK")),
                                "TOV": to_int(r.get("TOV")),
                                "PF": to_int(r.get("PF")),
                                "PTS": to_int(r.get("PTS")),
                                "PLUS_MINUS": None,
                                "VIDEO_AVAILABLE": None,
                                "PLAYER_NAME": player_name,
                                "SEASON_STRING": season_string,
                                "DATA_SOURCE": "sports-reference-cbb",
                            }
                        )

                    time.sleep(0.8)

        status = "ok" if player_rows else f"unmatched:{match.note if match else 'unknown'}"
        if status != "ok":
            unmatched.append(
                {
                    "PERSON_ID": player_id,
                    "PLAYER_NAME": player_name,
                    "DRAFT_YEAR": draft_year,
                    "reason": status,
                }
            )

        append_rows(OUT_PATH, player_rows)
        mark_progress(player_id, player_name, draft_year, status, len(player_rows))

        if (i + 1) % 25 == 0:
            print(f"Processed {i+1}/{len(cohort)} | last={player_name} | rows_written={len(player_rows)}")

        time.sleep(1.0)

    if unmatched:
        u = pd.DataFrame(unmatched)
        if UNMATCHED_PATH.exists():
            old = pd.read_csv(UNMATCHED_PATH)
            u = pd.concat([old, u], ignore_index=True)
        u = u.drop_duplicates(subset=["PERSON_ID"], keep="last")
        u.to_csv(UNMATCHED_PATH, index=False)

    print(f"Done. Data: {OUT_PATH}")
    print(f"Progress: {PROGRESS_PATH}")
    print(f"Unmatched: {UNMATCHED_PATH}")


if __name__ == "__main__":
    main()
