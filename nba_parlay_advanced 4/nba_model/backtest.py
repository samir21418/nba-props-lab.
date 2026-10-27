"""Leakage-controlled walk-forward evaluation of probability calibration.
Backtests hit probabilities at fixed thresholds; NOT betting ROI, because real
historical sportsbook odds are required for that claim.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .model import prepare_logs,dvp_ranking,simulate_matchup,_calibrated

TARGETS={'points':[14.5,19.5,24.5], 'rebounds':[5.5,7.5,9.5],
         'assists':[3.5,5.5,7.5],'threes':[1.5,2.5,3.5]}


def backtest(logs, minimum_history_days=28, max_days=12, simulations=2400):
    logs=logs.copy()
    logs['date']=pd.to_datetime(logs.date)
    days=sorted(logs.date.dt.date.unique())
    if not days:return pd.DataFrame()
    eligible=[d for d in days if (d-days[0]).days >= minimum_history_days]
    eligible=eligible[-max_days:]
    rows=[]
    for day in eligible:
        prior=prepare_logs(logs,day,rosters=None)
        dvp=dvp_ranking(prior)
        # This model does not observe future rosters/injuries; past-team membership only.
        todays=logs[logs.date.dt.date==day]
        for game_id,g in todays.groupby('game_id'):
            teams=list(g.team.dropna().unique())
            if len(teams)!=2:continue
            game={'game_id':str(game_id),'home':teams[0],'away':teams[1],'day':str(day)}
            sim,proj,meta=simulate_matchup(game,prior,dvp,rosters=None,n=simulations,
                                             rng=np.random.default_rng(12345))
            for _,actual in g.iterrows():
                pid=str(actual.player_id)
                if pid not in meta:continue
                for stat,thresholds in TARGETS.items():
                    for threshold in thresholds:
                        p=_calibrated(np.mean(sim[(pid,stat)]>threshold),meta[pid]['games'])
                        result=int(float(actual[stat])>threshold)
                        rows.append(dict(date=str(day),game_id=game_id,player_id=pid,stat=stat,
                                         line=threshold,prob=round(float(p),5),hit=result,
                                         brier=round(float((p-result)**2),6)))
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description='Evaluate NBA prop probability accuracy without future-data leakage')
    parser.add_argument('--input',default='data/game_logs.csv')
    parser.add_argument('--out',default='reports')
    parser.add_argument('--max-days',type=int,default=12)
    parser.add_argument('--min-history-days',type=int,default=28)
    opts=parser.parse_args()
    logs=pd.read_csv(opts.input,dtype={'player_id':str,'game_id':str})
    result=backtest(logs,max_days=opts.max_days,minimum_history_days=opts.min_history_days)
    out=Path(opts.out);out.mkdir(exist_ok=True,parents=True)
    result.to_csv(out/'backtest_predictions.csv',index=False)
    summary={}
    if not result.empty:
        for stat,g in result.groupby('stat'):
            summary[stat]={'examples':int(len(g)),'brier_score':round(float(g.brier.mean()),4),
                           'model_mean_probability':round(float(g.prob.mean()),3),
                           'observed_hit_rate':round(float(g.hit.mean()),3)}
        summary['overall']={'examples':int(len(result)),'brier_score':round(float(result.brier.mean()),4)}
    summary['notice']='Exploratory walk-forward probability test only. No historical odds/fees/ROI, and no proof of good calibration or profitability.'
    (out/'backtest_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
