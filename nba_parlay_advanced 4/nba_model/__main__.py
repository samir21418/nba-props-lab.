"""Command-line entry point: python -m nba_model --help"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from .sources import (schedule, get_rosters, get_injuries, update_logs, read_odds, load_odds_csv, ODDS_COLUMNS)
from .model import (prepare_logs,dvp_ranking,simulate_matchup,score_game,build_ladder,parlay_candidates)
from .report import generate_report

BASE=Path(__file__).resolve().parent.parent
LOG=logging.getLogger('nba-model')


def run(args):
    date=dt.datetime.now(ZoneInfo('America/Chicago')).date() if args.date=='today' else dt.date.fromisoformat(args.date)
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    warnings=[]
    if args.demo:
        from .demo import generate_demo
        games,logs,rosters,odds=generate_demo(date)
        warnings.append('DEMONSTRATION USING ENTIRELY SYNTHETIC PLAYERS AND HISTORICAL GAMES. No actual plays or prices.')
        injuries=pd.DataFrame()
    else:
        data=BASE/'data'
        log_path=data/'game_logs.csv'
        roster_path=data/'rosters.csv'
        if not args.no_update:
            n,errs=update_logs(log_path,history_days=args.history_days,seed_previous=args.seed_previous,today=date)
            warnings.extend(errs[:4]);LOG.info('Downloaded %s player-game stat rows',n)
            rosters=get_rosters()
            if not rosters.empty:rosters.to_csv(roster_path,index=False)
        if not log_path.exists():raise SystemExit('No game logs: run without --no-update to fetch ESPN data.')
        logs=pd.read_csv(log_path,dtype={'player_id':str,'game_id':str})
        rosters=pd.read_csv(roster_path,dtype={'player_id':str}) if roster_path.exists() else pd.DataFrame()
        if rosters.empty:warnings.append('Current rosters unavailable: positions/team assignments may be stale; verify before using.')
        games=schedule(date)
        try:
            injuries=get_injuries()
            injuries.to_csv(data/'injuries.csv',index=False)
        except Exception as e:
            warnings.append(f'Injury feed unavailable: {e}. Manually confirm starting lineups and availability.')
            injuries=pd.read_csv(data/'injuries.csv',dtype={'player_id':str}) if (data/'injuries.csv').exists() else pd.DataFrame()
            if not injuries.empty:warnings.append('Using cached injuries; may be stale.')
        key=os.environ.get('THE_ODDS_API_KEY','').strip()
        odds,errs=read_odds(key,date)
        warnings+=errs[:3]
        manual=load_odds_csv(data/'manual_odds.csv')
        if not manual.empty:
            manual=manual[manual.game_date.astype(str)==date.isoformat()]
            odds=pd.concat([odds,manual],ignore_index=True)
        if odds.empty:warnings.append('No sportsbook prop prices were loaded. Showing projected watchlists and unpriced ladder levels only.')
    hist=prepare_logs(logs,date,rosters)
    dvp=dvp_ranking(hist)
    if hist.empty:warnings.append('Insufficient historical player data for simulations.')
    if len(dvp.team.unique())<30 if not dvp.empty else True:
        warnings.append('DVP rankings cover only teams observed in historical source data; not necessarily the full 30-team league.')
    if args.demo and not hist.empty:
        LOG.warning('Synthetic demonstration. Do not use for wagering.')
    exclusions=set()
    if not injuries.empty:
        statuses=injuries.status.astype(str).str.lower()
        mask=statuses.str.contains(r'out|doubtful|questionable|game.time|inactive|suspended',regex=True,na=False)
        exclusions=set(injuries.loc[mask,'player_id'].astype(str))
        if exclusions:warnings.append(f'{len(exclusions)} listed injured/questionable players excluded from candidates; no teammate usage redistribution is modeled.')
    reports=[];all_picks=[];all_sims={};ladders=[]
    for game in games:
        if game['completed'] or game['state']!='pre':continue
        rng=np.random.default_rng(int(str(game['game_id'])[-7:]) if str(game['game_id'])[-7:].isdigit() else 714)
        sims,projections,metadata=simulate_matchup(game,hist,dvp,rosters,n=args.simulations,rng=rng,exclusions=exclusions)
        if not sims:
            warnings.append(f'Not enough historic player data for {game["away"]} @ {game["home"]}.')
        picks,watch=score_game(game,sims,projections,metadata,dvp,odds,date,min_edge=args.min_edge)
        ladder=build_ladder(game,sims,projections,metadata,odds)
        if ladder:ladders.append({'game':f"{game['away']} @ {game['home']}",**ladder})
        all_picks.extend(picks)
        all_sims.update(sims)
        reports.append({'game':game,'picks':picks,'watch':watch,'ladder':ladder})
    parlays=parlay_candidates(all_picks,all_sims)
    timestamp=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    report=generate_report(out/'index.html',reports,dvp,all_picks,parlays,warnings,date,timestamp)
    dvp.to_csv(out/'defense_vs_position_rankings.csv',index=False)
    pd.DataFrame(all_picks).to_csv(out/'qualified_props.csv',index=False)
    (out/'ladders.json').write_text(json.dumps(ladders,indent=2),encoding='utf-8')
    (out/'parlays.json').write_text(json.dumps(parlays,indent=2),encoding='utf-8')
    (out/'status.json').write_text(json.dumps({'date':date.isoformat(),'generated_utc':timestamp,
        'demo':args.demo,'games':len(reports),'qualified_picks':len(all_picks),
        'parlays':len(parlays),'warnings':warnings},indent=2),encoding='utf-8')
    print(f'Dashboard: {report}\nGames: {len(reports)} | priced qualifying picks: {len(all_picks)} | parlays: {len(parlays)}')
    for warning in warnings:print('NOTICE:',warning)


def main():
    p=argparse.ArgumentParser(description='Daily NBA prop, DVP and ladder analytical model')
    p.add_argument('--date',default='today',help='America/Chicago YYYY-MM-DD, or today')
    p.add_argument('--out',default='reports',help='Dashboard/report output directory')
    p.add_argument('--demo',action='store_true',help='Generate labeled synthetic demonstration with no network')
    p.add_argument('--no-update',action='store_true',help='Use existing local data (still loads schedule and odds)')
    p.add_argument('--history-days',type=int,default=35,help='Days to fetch on initial run (default 35)')
    p.add_argument('--seed-previous',action='store_true',help='On first run, add preceding spring for early-season historical baseline; slow')
    p.add_argument('--simulations',type=int,default=14000,help='Monte Carlo sample count')
    p.add_argument('--min-edge',type=float,default=.045,help='Min model estimated probability edge over breakeven')
    args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')
    run(args)

if __name__=='__main__':main()
