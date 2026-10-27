"""HTTP source adapters. ESPN endpoints are public but unofficial and can change."""
from __future__ import annotations
import datetime as dt
import logging
import os
import re
import time
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import requests

LOG = logging.getLogger(__name__)
ESPN = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba'
ODDS = 'https://api.the-odds-api.com/v4/sports/basketball_nba'
MARKETS = {'player_points':'points','player_rebounds':'rebounds','player_assists':'assists',
           'player_threes':'threes'}
ALT_MARKETS = {m+'_alternate':s for m,s in MARKETS.items()}
STATS = {'MIN':'minutes','PTS':'points','REB':'rebounds','AST':'assists','3PT':'threes'}
POSITIONS = ('PG','SG','SF','PF','C')
CENTRAL = ZoneInfo('America/Chicago')
HEADERS = {'User-Agent':'Mozilla/5.0 (compatible; personal-NBA-analysis/1.0)','Accept':'application/json'}


def get_json(url, params=None, session=None, retries=3):
    client = session or requests
    for attempt in range(retries):
        try:
            resp = client.get(url, params=params, headers=HEADERS, timeout=25)
            if resp.status_code == 429 and attempt < retries-1:
                time.sleep(5*(attempt+1)); continue
            if resp.status_code in (403, 429):
                raise RuntimeError(f'API returned {resp.status_code}: {url}. Check service access and quotas.')
            resp.raise_for_status()
            return resp.json()
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError):
            if attempt == retries-1: raise
            time.sleep(1.5 * (attempt+1))


def schedule(day: dt.date, session=None):
    payload = get_json(f'{ESPN}/scoreboard', {'dates':day.strftime('%Y%m%d'), 'limit':'100'}, session)
    results = []
    for event in payload.get('events', []):
        # No preseason/playoffs mixed in regular season model.
        if event.get('season',{}).get('type',2) != 2: continue
        comps=event.get('competitions',[])
        if not comps: continue
        teams={c.get('homeAway'):c.get('team',{}).get('abbreviation') for c in comps[0].get('competitors',[])}
        if not all(teams.get(s) for s in ('home','away')): continue
        start=event.get('date','')
        status=event.get('status',{}).get('type',{})
        results.append(dict(game_id=str(event.get('id')), day=day.isoformat(), start_utc=start,
                            home=teams['home'], away=teams['away'],
                            state=status.get('state','pre'), completed=bool(status.get('completed'))))
    return results


def _num(s):
    try: return float(str(s).replace(',',''))
    except (TypeError, ValueError): return 0.0


def _minutes(s):
    if ':' in str(s):
        bits=str(s).split(':')
        return _num(bits[0])+_num(bits[1])/60.0
    return _num(s)


def _made(s):
    """3PT '3-8' -> 3; ESPN boxscore uses makes-attempts."""
    return _num(str(s).split('-')[0])


def parse_boxscore(data, game):
    """Parse ESPN summary boxscore players, supporting top-level player blocks."""
    box=data.get('boxscore',{})
    groups=box.get('players',[])
    if not groups:
        for team_block in box.get('teams',[]):
            groups.extend(team_block.get('players',[]))
    records=[]
    for group in groups:
        team=group.get('team',{}).get('abbreviation','')
        if not team:
            team_id=str(group.get('team',{}).get('id',''))
            # fallback to ESPN header id mappings
            for c in data.get('header',{}).get('competitions',[{}])[0].get('competitors',[]):
                if str(c.get('team',{}).get('id',c.get('id'))) == team_id:
                    team=c.get('team',{}).get('abbreviation',''); break
        if team not in (game['home'],game['away']):
            LOG.warning('Skipped boxscore group with unmatched team %s for %s',team,game['game_id'])
            continue
        opponent=game['away'] if team==game['home'] else game['home']
        for stat_group in group.get('statistics',[]):
            names=[str(x).upper() for x in stat_group.get('names',stat_group.get('labels',[]))]
            for row in stat_group.get('athletes',[]):
                if row.get('didNotPlay') or row.get('ejected') and not row.get('stats'): continue
                athlete=row.get('athlete',{})
                raw=dict(zip(names,row.get('stats',[])))
                minutes=_minutes(raw.get('MIN',0))
                if minutes < 1: continue
                p=athlete.get('position',{}) or {}
                position=p.get('abbreviation','') if isinstance(p,dict) else str(p)
                stat=dict(minutes=minutes,points=_num(raw.get('PTS')),rebounds=_num(raw.get('REB')),
                          assists=_num(raw.get('AST')),threes=_made(raw.get('3PT','0')))
                records.append(dict(date=game['day'],game_id=game['game_id'],player_id=str(athlete.get('id','')),
                                    player=athlete.get('displayName',''),team=team,opponent=opponent,
                                    position=position,**stat))
    if not records: raise ValueError(f'No usable stats from ESPN summary for game {game["game_id"]}')
    return records


def get_rosters(session=None):
    """Current roster is used to resolve positions and avoid recommending traded players."""
    team_data=get_json(f'{ESPN}/teams', {'limit':'100'}, session)
    teams=[]
    for sport in team_data.get('sports',[]):
        for league in sport.get('leagues',[]):
            teams += [x.get('team',{}) for x in league.get('teams',[])]
    players=[]
    for team in teams:
        tid=team.get('id'); abbr=team.get('abbreviation','')
        try:
            roster=get_json(f'{ESPN}/teams/{tid}/roster',session=session)
        except (requests.RequestException,RuntimeError):
            LOG.warning('Roster unavailable: %s',abbr); continue
        # Both grouped and ungrouped formats have been seen.
        for group in roster.get('athletes',[]):
            for athlete in group.get('items',[group]):
                pos=athlete.get('position',{}) or {}
                if isinstance(pos,dict): pos=pos.get('abbreviation','')
                players.append(dict(player_id=str(athlete.get('id','')),player=athlete.get('displayName',''),
                                    team=abbr,position=pos))
        time.sleep(.03)
    return pd.DataFrame(players,columns=['player_id','player','team','position']).drop_duplicates('player_id')


def update_logs(path, history_days=25, seed_previous=False, today=None, session=None):
    """Incremental ESPN ingestion. On first run, backfills last history_days calendar days.
    Optionally seed preceding spring 45 days to support early-season projections.
    Existing CSV data is preserved and dates are appended, never invented.
    """
    today=today or dt.datetime.now(CENTRAL).date()
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        old=pd.read_csv(path,dtype={'player_id':str,'game_id':str})
        if not old.empty:
            # refresh last 3 days for corrections; avoid refetching all history each run.
            begin=max(dt.date.fromisoformat(old['date'].max())-dt.timedelta(days=3),today-dt.timedelta(days=history_days))
        else: begin=today-dt.timedelta(days=history_days)
    else:
        old=pd.DataFrame();begin=today-dt.timedelta(days=history_days)
    dates=[]; day=begin
    while day<today:
        dates.append(day);day+=dt.timedelta(days=1)
    if seed_previous and old.empty:
        # Last regular-season spring as a warm start. Includes non-NBA dates harmlessly.
        spring_year=today.year if today.month >= 7 else today.year-1
        spring_end=dt.date(spring_year,4,17)
        spring_start=spring_end-dt.timedelta(days=46)
        dates = [spring_start+dt.timedelta(days=i) for i in range((spring_end-spring_start).days+1)] + dates
    dates=sorted(set(dates))
    updated=[]; errors=[]; seen=set()
    for date in dates:
        try:
            events=schedule(date,session)
        except (requests.RequestException,RuntimeError) as e:
            errors.append(f'Scoreboard {date}: {e}'); continue
        for e in events:
            if not e['completed'] or e['game_id'] in seen: continue
            seen.add(e['game_id'])
            try:
                updated.extend(parse_boxscore(get_json(f'{ESPN}/summary', {'event':e['game_id']},session),e))
                time.sleep(.8)  # ESPN does not publish a rate limit: conservative pacing
            except (ValueError, requests.RequestException,RuntimeError,IndexError) as ex:
                errors.append(f'Boxscore {e["game_id"]}: {ex}')
        time.sleep(.2)
    combined=pd.concat([old,pd.DataFrame(updated)],ignore_index=True)
    if combined.empty: raise RuntimeError('No historical logs available. ESPN data could not be downloaded.')
    combined=combined.drop_duplicates(['game_id','player_id'],keep='last')
    combined.to_csv(path,index=False)
    return len(updated),errors


def read_odds(api_key, day, session=None):
    """The Odds API v4 per-event props. No key => no market-price claims."""
    if not api_key: return pd.DataFrame(columns=ODDS_COLUMNS),['No THE_ODDS_API_KEY: projections only (not value bets)']
    errs=[];rows=[]
    params={'apiKey':api_key,'regions':'us','dateFormat':'iso'}
    try:
        events=get_json(f'{ODDS}/events',params,session)
    except (requests.RequestException,RuntimeError) as e:
        return pd.DataFrame(columns=ODDS_COLUMNS),[f'Odds events not available: {e}']
    for event in events:
        try:
            start=dt.datetime.fromisoformat(event['commence_time'].replace('Z','+00:00')).astimezone(CENTRAL).date()
            if start != day: continue
            params2={'apiKey':api_key,'regions':'us','oddsFormat':'american',
                     'markets':','.join(list(MARKETS)+list(ALT_MARKETS))}
            detail=get_json(f'{ODDS}/events/{event["id"]}/odds',params2,session)
            for b in detail.get('bookmakers',[]):
                for m in b.get('markets',[]):
                    key=m.get('key','')
                    if key not in MARKETS and key not in ALT_MARKETS: continue
                    stat=MARKETS.get(key,ALT_MARKETS.get(key))
                    for x in m.get('outcomes',[]):
                        if x.get('name') not in ('Over','Under') or not x.get('description'): continue
                        if not isinstance(x.get('point'),(int,float)):continue
                        try:price=int(x['price'])
                        except (ValueError,TypeError):continue
                        rows.append(dict(game_date=day.isoformat(),home=detail.get('home_team',event.get('home_team','')),
                                         away=detail.get('away_team',event.get('away_team','')),
                                         player=x['description'],stat=stat,side=x['name'],line=float(x['point']),
                                         price=price,book=b.get('title',''),alternate=(key in ALT_MARKETS),
                                         last_update=m.get('last_update',b.get('last_update',''))))
        except (requests.RequestException,RuntimeError) as e:
            errs.append(f'Odds for {event.get("id")}: {e}')
        time.sleep(.05)
    return pd.DataFrame(rows,columns=ODDS_COLUMNS),errs

ODDS_COLUMNS=['game_date','home','away','player','stat','side','line','price','book','alternate','last_update']

# Map provider naming differences; add aliases if odds vendor uses another spelling.
ALIASES={'LA Clippers':'LAC','Los Angeles Clippers':'LAC','LA Lakers':'LAL',
         'Los Angeles Lakers':'LAL','Golden State Warriors':'GSW','New York Knicks':'NYK',
         'Brooklyn Nets':'BKN','San Antonio Spurs':'SAS','New Orleans Pelicans':'NOP',
         'Portland Trail Blazers':'POR','Philadelphia 76ers':'PHI','Oklahoma City Thunder':'OKC',
         'Boston Celtics':'BOS','Chicago Bulls':'CHI','Cleveland Cavaliers':'CLE',
         'Dallas Mavericks':'DAL','Denver Nuggets':'DEN','Detroit Pistons':'DET',
         'Houston Rockets':'HOU','Indiana Pacers':'IND','Memphis Grizzlies':'MEM',
         'Miami Heat':'MIA','Milwaukee Bucks':'MIL','Minnesota Timberwolves':'MIN',
         'Orlando Magic':'ORL','Phoenix Suns':'PHX','Sacramento Kings':'SAC',
         'Toronto Raptors':'TOR','Utah Jazz':'UTA','Washington Wizards':'WAS',
         'Atlanta Hawks':'ATL','Charlotte Hornets':'CHA'}

def map_team(v):return ALIASES.get(str(v),str(v))

def normal_name(x):
    return re.sub(r'[^a-z0-9]','',str(x).lower())


def load_odds_csv(file):
    if not Path(file).exists():return pd.DataFrame(columns=ODDS_COLUMNS)
    df=pd.read_csv(file)
    # user may specify alternative manually; only real recorded prices should be entered
    for col in ODDS_COLUMNS:
        if col not in df.columns: raise ValueError(f'Manual odds missing column: {col}')
    return df


def get_injuries(session=None):
    """Unconfirmed ESPN injury feed, not an official availability determination.
    Only a correctly retrieved response is returned; callers must show missing-feed warning.
    """
    payload=get_json(f'{ESPN}/injuries',session=session)
    rows=[]
    for item in payload.get('injuries',[]):
        athlete=item.get('athlete',{}) or {}
        status=item.get('status','')
        if isinstance(status,dict):status=status.get('name',status.get('displayName',''))
        if athlete.get('id'):
            rows.append({'player_id':str(athlete['id']), 'player':athlete.get('displayName',''),
                         'status':str(status),'reported_at':str(item.get('date',''))})
    return pd.DataFrame(rows,columns=['player_id','player','status','reported_at'])
