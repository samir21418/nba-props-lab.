# NBA Props Lab — Advanced Python Model

**Status:** Working, locally tested prototype. Not yet deployed and **not yet supplied with live NBA statistics or betting odds** in this download. The included HTML demonstration uses **entirely fabricated players, data and team codes**; it must not be used as NBA betting advice.

## What the project delivers

- **Defense vs. position (DVP):** five primary roster positions, PG/SG/SF/PF/C; 4 stats (points, rebounds, assists, threes); matchup allowances **per 36 minutes**; rankings across all teams present in the ingested game log. Rank **1 = most generous defense** for that position/stat, not toughest defense. Partial-season samples are shrunk toward the league mean. Downloadable CSV and interactive HTML filters.
- **For every upcoming game:** up to three sportsbook **value candidates** where model-estimated probability is at least 54%, exceeds quoted breakeven probability by at least 4.5 percentage points, and has positive estimated expected value. **No forced bets** if fewer than 2–3 pass. Otherwise, three explicitly *unpriced watchlist* players/markets appear.
- **One ladder candidate per game:** three ascending milestones for one player. If alternate sportsbook odds are available, the model shows those quoted prices; otherwise, it displays **model-only** milestones and theoretical fair prices. A ladder comprises *separate* single bets, **not** a 3-leg parlay. Price verification alone is not an edge claim.
- **2- and 3-leg parlays:** Monte Carlo joint-event estimates, payout and estimated profit at a $10 stake; no simple product-of-leg-probabilities assumption. Same-game shared pace and same-player stat sampling are approximations, not fully fitted joint distributions.
- **Injuries:** tries the ESPN NBA injury feed daily and excludes players listed Out, Doubtful, Questionable, or Game-Time Decision. Does **not** automatically reallocate injured players' usage or verify starting lineups. Always check the official NBA injury report and confirmed lineups.
- **Automatic schedules:** included GitHub Actions workflow at **11:17 AM and 5:17 PM America/Chicago time**, with a GitHub Pages HTML publishing step. This schedule does not activate until the repository is configured on your GitHub account.
- **Backtesting:** past-only, walk-forward probability assessment, Brier scores and mean probabilities versus actual hit rates on standard threshold tests. This is **not** betting ROI or proof of calibration because it does not contain historical closing prop prices.

## Quick start on your laptop

You need **Python 3.11+** and an internet connection for non-demo runs.

```bash
cd nba_parlay_advanced
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

### Look at the offline example first

```bash
python -m nba_model --demo --date 2026-10-27 --out demo_report
```

Open `demo_report/index.html`. The sample players are named `FAKE ...` and the games **do not represent real games**, even though the demonstration date is October 27.

### Run a real daily report

The Odds API offers NBA player prop markets, but **player props and alternate lines may require a paid subscription**, and event-level calls consume API usage credits. Get your own key at https://the-odds-api.com/ and check its plan permissions for `player_points`, `player_rebounds`, `player_assists`, `player_threes`, and the `_alternate` markets.

```bash
export THE_ODDS_API_KEY="YOUR_PRIVATE_KEY"  # On Windows use: set THE_ODDS_API_KEY=...
python -m nba_model --seed-previous --out reports
```

Open `reports/index.html` or upload the project to GitHub for publishing. Initial run backfills the previous 35 calendar days **plus a late-previous-season sample** when `--seed-previous` is specified. Initial ingestion can take several minutes and ESPN may throttle/block cloud machines. Later runs append/update recent box scores. A season-accumulating dataset is more reliable than an early-season small sample.

Without an odds key:

```bash
python -m nba_model --seed-previous --out reports
```

...still creates positional rankings, player projections, game watchlists, and theoretical unpriced ladder thresholds, **but will generate no price-backed recommended plays or priced parlays**.

### Manual alternative for sportsbooks/lines

Copy `data/manual_odds_TEMPLATE.csv` to `data/manual_odds.csv`, then fill the fields using **real** sportsbook quotes:

| Column | Sample format |
|---|---|
| `game_date` | `2026-10-27` |
| `home`, `away` | Full NBA team names e.g. `Boston Celtics` |
| `player` | Full player name (matching ESPN) |
| `stat` | `points`, `rebounds`, `assists`, `threes` |
| `side` | `Over` or `Under` |
| `line` | `24.5` |
| `price` | American signed integer e.g. `-110` or `125` |
| `book` | Sportsbook name |
| `alternate` | `False` for standard props; `True` for alternate ladders |
| `last_update` | ISO-8601 UTC timestamp, e.g. `2026-10-27T19:00:00Z` |

Current-day quotes older than 8 hours, or without a readable timestamp, are **not treated as verified**. Manual entries are trusted as accurate input; the program cannot independently verify typed-in sportsbook prices.

## Make it update automatically and be viewable on your phone

1. Make a **new GitHub repository** and upload this entire extracted folder, **including `.github/workflows/update-daily.yml`**. Avoid committing `.env` files or API keys.
2. In the repository's **Settings → Secrets and variables → Actions → New repository secret**, add `THE_ODDS_API_KEY`. You can omit it, but you'll get only model-only watchlists.
3. In **Settings → Pages**, select **GitHub Actions** as the build/deployment source. Check the repository's GitHub Pages access/visibility policies.
4. In **Actions → Daily NBA model → Run workflow**, start a manual trial. Review logs, Pages URL, data quality and book prices. Thereafter the workflow is scheduled for 11:17 AM and 5:17 PM **America/Chicago** daily. GitHub scheduled workflows can be delayed or disabled and are not a guaranteed timing service.
5. Open the deployed Pages URL from your iPhone. It shows games, plays, ladders, DVP filters and parlays. The repo also retains daily CSV and JSON outputs.

**Important:** This environment cannot confirm whether ESPN will allow the first live collection, and I cannot deploy or activate your GitHub Action by packaging this project. The mock tests validate code on known-shaped fixtures, not end-to-end access to live services.

## Files and outputs

```text
nba_model/sources.py           # ESPN and The Odds API adapters
nba_model/model.py             # projections, DVP rankings and correlated simulation
nba_model/report.py            # self-contained HTML dashboard
nba_model/backtest.py          # walk-forward Brier-score backtest
nba_model/demo.py              # synthetic offline example
nba_model/__main__.py          # daily CLI
.github/workflows/update-daily.yml # scheduled update + Pages deployment
examples/demo_dashboard.html  # fictional example interface
requirements.txt
reports/                       # created by the real run
```

Real runs write `reports/index.html`, `defense_vs_position_rankings.csv`, `qualified_props.csv`, `ladders.json`, `parlays.json`, and `status.json`.

Run a past-data check when enough genuine game logs have accumulated:

```bash
python -m nba_model.backtest --input data/game_logs.csv --out reports --max-days 12
```

### How the probability model works

1. Reads past-only game logs, excludes known availability risks from the candidate list, and estimates expected minutes and per-minute production using weighted recent games.
2. Calculates per-36 DVP stats by each player's listed **primary** roster position, with partial pooling toward a league average. This is **not** the same as tracking actual positional matchup possessions.
3. Adjusts baseline stats by a capped DVP ratio (+/- ~15%), then runs a bootstrap-style simulation with minute volatility and a shared game-pace factor. It retains correlated outcomes between a player's stats by sampling from a common past game.
4. Applies a heuristic, conservative shrink of modeled hit probabilities toward 50%. **This shrink is NOT statistically calibrated**; real calibration must come from out-of-sample history.
5. Joins exact sportsbook markets/player/line/side, computes break-even probabilities from American odds, screens for minimum estimated margin and ranks candidate combinations.

**Risks and missing factors:** The approach is exploratory. It does not yet rigorously model possession-level matchups, game spread/blowout, pace shifts from projected rotations, confirmed starting lineups, usage changes due to injuries, actual conditional correlation of all players in a game, or historical line closing value. It can overfit a small sample; model EV can be positive while true EV is negative. You should validate on a large out-of-sample dataset before risking money. A single rung or candidate is *not* promised to win, and requiring two or three picks per game regardless of their estimated edge is a poor betting discipline.

## API references (checked October 2026)

- ESPN site-v2 NBA scoreboard, boxscore summaries, teams/rosters, injuries: https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard (public but unofficial; no availability guarantees).
- The Odds API V4 event odds and alternate NBA player prop markets: https://the-odds-api.com/liveapi/guides/v4/ and https://the-odds-api.com/sports-odds-data/betting-markets.html.
- GitHub Actions scheduling/timezones: https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#onschedule.

*No money is transferred, no bets are placed, and no implied results are guaranteed.*
