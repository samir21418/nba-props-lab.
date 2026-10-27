"""Fully fictional synthetic fixtures for exercising the report offline."""
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd


def generate_demo(day):
    rng=np.random.default_rng(113)
    pairs=[('AA1','BB1'),('CC2','DD2'),('EE3','FF3'),('GG4','HH4')]
    teams=[x for pair in pairs for x in pair]
    positions=['PG','SG','SF','PF','C']
    games=[{'home':home,'away':away,'day':day.isoformat(),'game_id':str(20000+i),'state':'pre','completed':False} for i,(home,away) in enumerate(pairs)]
    # Round-robin fixture IDs: both teams share a game ID for real backtest structure.
    fixtures={}
    for g in range(23):
        rotation=g%7
        order=[teams[0]]+teams[1+rotation:]+teams[1:1+rotation]
        for i in range(4):
            a,b=order[i],order[-1-i]
            gid=f'SYN-{g}-{min(a,b)}-{max(a,b)}'
            fixtures[(g,a)]=(gid,b)
            fixtures[(g,b)]=(gid,a)
    logs=[];rosters=[]
    for t in teams:
        for ix,pos in enumerate(positions):
            player=f'FAKE {t} {pos}'
            pid=f'{t}{ix}'
            rosters.append({'player_id':pid,'team':t,'player':player,'position':pos})
            for g in range(23):
                past_date=day-dt.timedelta(days=34-g)
                gid,opponent=fixtures[(g,t)]
                m=np.clip(rng.normal(31 if ix<3 else 26,3),12,42)
                p=max(0,int(rng.normal(22 if ix<2 else 14,5)))
                r=max(0,int(rng.normal(7 if ix>2 else 4,2)))
                a=max(0,int(rng.normal(7 if ix==0 else 3,2)))
                th=max(0,int(rng.normal(3 if ix<3 else 1.1,1.1)))
                logs.append(dict(date=past_date.isoformat(),game_id=gid,player_id=pid,
                                 player=player,team=t,opponent=opponent,position=pos,
                                 minutes=round(float(m),1),points=p,rebounds=r,assists=a,threes=th))
    from .sources import ODDS_COLUMNS
    return games,pd.DataFrame(logs),pd.DataFrame(rosters),pd.DataFrame(columns=ODDS_COLUMNS)
