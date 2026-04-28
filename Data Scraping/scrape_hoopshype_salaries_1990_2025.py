import csv
import json
import time
from pathlib import Path
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.hoopshype.com"
OUT_DIR = Path("data/hoopshype_salaries")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = list(range(1990, 2026))

QUERY = """
query ContractsSalariesPlayersIndexPage($season: Int, $size: Int, $salaryGreaterThanEq: Int, $salaryLessThanEq: Int, $cursor: String) {
  __typename
  contracts(
    season: $season
    size: $size
    salaryGreaterThanEq: $salaryGreaterThanEq
    salaryLessThanEq: $salaryLessThanEq
    cursor: $cursor
  ) {
    __typename
    numResults
    cursor
    contracts {
      __typename
      ...ContractsSalariesPlayersIndexFragment
    }
  }
}
fragment ContractsSalariesPlayersIndexFragment on Contracts {
  __typename
  playerID
  playerName
  updateDate
  player {
    __typename
    id
    firstName
    lastName
  }
  seasons {
    __typename
    teamOption
    playerOption
    qualifyingOffer
    twoWayContract
    salary
    season
    teamID
    capAllocation
    notes
    terminated
  }
}
""".strip()

HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "X-Api-Type": "sports2",
    "X-SiteCode": "USAT",
    "User-Agent": "Mozilla/5.0",
    "Referer": f"{BASE_URL}/salaries/players/",
}


def fetch_next_data_page(season: int):
    url = f"{BASE_URL}/salaries/players/?season={season}"
    html = requests.get(url, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")
    next_data = soup.find("script", id="__NEXT_DATA__")
    if not next_data:
        raise RuntimeError(f"No __NEXT_DATA__ on season {season}")
    payload = json.loads(next_data.string)
    query_blobs = payload["props"]["pageProps"]["dehydratedState"]["queries"]
    season_blob = next((q for q in query_blobs if q.get("queryKey") == [season]), None)
    if not season_blob:
        raise RuntimeError(f"No season blob found in __NEXT_DATA__ for {season}")
    contracts_obj = season_blob["state"]["data"]["pages"][0]["contracts"]
    return contracts_obj


def fetch_full_via_api(season: int):
    all_contracts = []
    cursor = None
    while True:
        params = {
            "query": QUERY,
            "variables": json.dumps({"season": season, "size": 20, "cursor": cursor}),
        }
        r = requests.get(f"{BASE_URL}/api/data/", params=params, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"API status {r.status_code} for season {season}: {r.text[:200]}")
        payload = r.json()
        if not payload or not payload.get("data"):
            raise RuntimeError(f"API empty/false payload for season {season}: {str(payload)[:300]}")

        contracts_obj = payload["data"]["contracts"]
        rows = contracts_obj.get("contracts") or []
        all_contracts.extend(rows)
        cursor = contracts_obj.get("cursor")
        if not cursor:
            break
        time.sleep(0.12)

    return all_contracts


def flatten_contracts(season: int, contracts: list[dict]):
    rows = []
    for c in contracts:
        player_id = c.get("playerID")
        player_name = c.get("playerName")
        update_date = c.get("updateDate")
        seasons = c.get("seasons") or []
        for s in seasons:
            if s.get("season") != season:
                continue
            rows.append(
                {
                    "season": season,
                    "player_id": player_id,
                    "player_name": player_name,
                    "team_id": s.get("teamID"),
                    "salary": s.get("salary"),
                    "cap_allocation": s.get("capAllocation"),
                    "team_option": s.get("teamOption"),
                    "player_option": s.get("playerOption"),
                    "qualifying_offer": s.get("qualifyingOffer"),
                    "two_way_contract": s.get("twoWayContract"),
                    "terminated": s.get("terminated"),
                    "notes": s.get("notes"),
                    "update_date": update_date,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]):
    fields = [
        "season",
        "player_id",
        "player_name",
        "team_id",
        "salary",
        "cap_allocation",
        "team_option",
        "player_option",
        "qualifying_offer",
        "two_way_contract",
        "terminated",
        "notes",
        "update_date",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    summary = []
    all_rows = []

    for season in YEARS:
        try:
            contracts = fetch_full_via_api(season)
            source = "api"
            expected = len(contracts)
        except Exception:
            fallback = fetch_next_data_page(season)
            contracts = fallback.get("contracts", [])
            source = "next_data_fallback_top20"
            expected = fallback.get("numResults")

        rows = flatten_contracts(season, contracts)
        all_rows.extend(rows)

        season_json = OUT_DIR / f"salaries_players_{season}.json"
        season_csv = OUT_DIR / f"salaries_players_{season}.csv"
        season_json.write_text(json.dumps(contracts, ensure_ascii=False, indent=2), encoding="utf-8")
        write_csv(season_csv, rows)

        summary.append(
            {
                "season": season,
                "source": source,
                "contracts_rows": len(contracts),
                "season_salary_rows": len(rows),
                "expected_num_results": expected,
            }
        )
        print(f"{season}: {source} contracts={len(contracts)} salary_rows={len(rows)} expected={expected}")

    write_csv(OUT_DIR / "salaries_players_1990_2025_combined.csv", all_rows)
    (OUT_DIR / "salaries_players_1990_2025_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Done. Combined rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
