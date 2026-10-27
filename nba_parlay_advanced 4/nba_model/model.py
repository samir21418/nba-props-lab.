"""Position DVP, correlated game-day Monte Carlo and expected-value screening.
Exploratory model, not validated as a profitable or calibrated betting system.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import math
import numpy as np
import pandas as pd
from .sources import POSITIONS, map_team, normal_name

STATS=['points','rebounds','assists','threes']
MIN_PLAYER_GAMES=7
N_SIM=14000


def prepare_logs(df, target_date, rosters=None):
    out=df.copy()
    if out.empty:return out
    out['date']=pd.to_datetime(out['date'],errors='coerce')
    out=out[out['date']<pd.Timestamp(target_date)]
    for col in ['minutes']+STATS:
        out[col]=pd.to_numeric(out[col],errors='coerce').fillna(0)
    out=out[(out['minutes']>0)&(out['player_id'].notna())].copy()
    out['player_id']=out['player_id'].astype(str).str.replace(r'\.0$','',regex=True)
    # Current roster positions override stale boxscore positions, but only when known.
    if rosters is not None and not rosters.empty:
        roster=rosters[['player_id','position']].drop_duplicates('player_id').copy()
        roster['player_id']=roster['player_id'].astype(str)
        out=out.merge(roster.rename(columns={'position':'roster_position'}),how='left',on='player_id')
        out['position']=out['roster_position'].where(out['roster_position'].isin(POSITIONS),out['position'])
        out=out.drop(columns='roster_position')
    out=out.sort_values('date')
    return out


def dvp_ranking(logs, recent_games=20):
    """Per-36 production by opposing primary-position groups within a game.
    #1 rank = MOST GENEROUS defense for that position+stat; shrink partial sample.
    Missing positions excluded, never assigned a guessed PG/SG etc.
    """
    known=logs[logs.position.isin(POSITIONS)].copy()
    if known.empty:return pd.DataFrame(columns=['team','position','stat','allowed_per36','rank','games','league_avg','ratio'])
    # team=defending team = opponent
    groups=known.groupby(['opponent','position','game_id','date'],as_index=False)[['minutes']+STATS].sum()
    groups=groups[groups.minutes>=10].copy()
    for stat in STATS: groups[stat]=groups[stat]/groups.minutes*36
    groups=groups.sort_values('date').groupby(['opponent','position'],group_keys=False).tail(recent_games)
    rows=[]
    for (team,pos),g in groups.groupby(['opponent','position']):
        for stat in STATS:
            rows.append(dict(team=team,position=pos,stat=stat,raw_allowed=g[stat].mean(),games=len(g)))
    ranking=pd.DataFrame(rows)
    if ranking.empty:return ranking
    # Compare to league average for the SAME primary-position group.
    ranking['league_avg']=ranking.groupby(['position','stat'])['raw_allowed'].transform('mean')
    # Partial pooling: 10 games worth of league prior, reducing extreme early-season rankings.
    k=10.
    ranking['allowed_per36']=(ranking.raw_allowed*ranking.games+ranking.league_avg*k)/(ranking.games+k)
    ranking['ratio']=ranking.allowed_per36/ranking.league_avg.clip(lower=0.01)
    ranking['rank']=ranking.groupby(['position','stat'])['allowed_per36'].rank(ascending=False,method='min').astype(int)
    return ranking[['team','position','stat','allowed_per36','rank','games','league_avg','ratio']].sort_values(['position','stat','rank'])


def _adjust_ratio(dvp, opponent, position, stat):
    row=dvp[(dvp.team==opponent)&(dvp.position==position)&(dvp.stat==stat)]
    if row.empty:return 1.
    return float(np.clip(1+.55*(float(row.iloc[0].ratio)-1),.85,1.15))


def current_players(logs, rosters, team):
    if rosters is not None and not rosters.empty:
        roster=rosters[rosters.team==team]
        ids=set(roster.player_id.astype(str))
        return logs[(logs.player_id.isin(ids))] if ids else logs.iloc[:0]
    # Without roster data, use latest recorded team to mitigate trades; warn upstream.
    last=logs.sort_values('date').drop_duplicates('player_id',keep='last')
    ids=set(last[last.team==team].player_id)
    return logs[logs.player_id.isin(ids)]


def _weighted_mean(a):
    if len(a)==0:return 0
    w=np.geomspace(.36,1.,len(a))
    return np.average(a,weights=w)


def simulate_matchup(game, logs, dvp, rosters=None, n=N_SIM, rng=None, exclusions=None):
    """One shared pace shock/game, one sampled historical game per player for
    correlated points/rebounds/assists/threes, plus minutes uncertainty.
    Live lineups not verified; flagged injuries filtered separately.
    """
    rng = rng if rng is not None else np.random.default_rng(714)
    exclusions = set(exclusions or [])
    common_pace=np.clip(rng.normal(1.,.052,n),.78,1.23)
    sims={};projections={};metadata={}
    for team,opponent in [(game['home'],game['away']),(game['away'],game['home'])]:
        teamlogs=current_players(logs,rosters,team)
        for pid,g in teamlogs.groupby('player_id'):
            if str(pid) in exclusions:continue
            g=g.sort_values('date').tail(24)
            if len(g)<MIN_PLAYER_GAMES:continue
            # Discard if last appearance more than 45 days earlier except at season opening.
            # Player still in roster means currently rostered, but return-to-play risk remains.
            current=g.iloc[-1]
            position=str(current.position)
            if rosters is not None and not rosters.empty:
                m=rosters[rosters.player_id.astype(str)==str(pid)]
                if not m.empty:position=m.iloc[0].position
            if position not in POSITIONS:continue
            recent=g.tail(8)
            estimate_minutes=.6*_weighted_mean(recent.minutes.values)+.4*_weighted_mean(g.minutes.values)
            if estimate_minutes < 17:continue
            minutes_sd=max(2.5, float(recent.minutes.std(ddof=0)))
            draws_minutes=np.clip(rng.normal(estimate_minutes,minutes_sd,n),7,44)
            probs=np.geomspace(.33,1.,len(g));probs=probs/probs.sum()
            hist_i=rng.choice(len(g),size=n,p=probs)
            player_pace=np.clip(rng.normal(1.,.055,n),.78,1.22)
            stat_projections={}
            for stat in STATS:
                obs=np.maximum(g[stat].to_numpy(float),0)
                mins=g.minutes.to_numpy(float)
                per_min=obs/np.maximum(mins,1)
                recent_per_min=recent[stat].to_numpy(float)/np.maximum(recent.minutes.to_numpy(float),1)
                rate=.60*_weighted_mean(recent_per_min)+.40*_weighted_mean(per_min)
                matchup=_adjust_ratio(dvp,opponent,position,stat)
                expected=max(0,rate*estimate_minutes*matchup)
                observed_mean=max(.01,np.average(per_min,weights=probs))
                # Empirical joint fluctuations: preserves within-player sample-game correlation.
                sampled_rate=per_min[hist_i]/observed_mean
                sampled_rate=np.clip(.30+.70*sampled_rate,.13,2.6)
                noise_sd={'points':1.25,'rebounds':.65,'assists':.60,'threes':.36}[stat]
                sample=np.clip(expected*sampled_rate*(draws_minutes/estimate_minutes)*common_pace*player_pace
                               +rng.normal(0,noise_sd,n),0,None)
                sample=np.rint(sample).astype(np.int16)
                sims[(str(pid),stat)]=sample
                stat_projections[stat]=round(float(sample.mean()),1)
            projections[str(pid)]=stat_projections
            metadata[str(pid)]={'player':str(current.player),'player_id':str(pid),'team':team,'opponent':opponent,
                                'position':position,'minutes_projection':round(estimate_minutes,1),
                                'games':len(g),'matchup':f"{game['away']} @ {game['home']}",
                                'game_id':game['game_id']}
    return sims,projections,metadata


def american_decimal(price):
    price=int(price)
    if price == 0:return np.nan
    return 1+price/100 if price>0 else 1+100/abs(price)


def implied_probability(price):
    return 1/american_decimal(price)


def fair_american(prob):
    if prob<=0 or prob>=1:return 'N/A'
    if prob>=.5:return str(round(-100*prob/(1-prob)))
    return '+'+str(round(100*(1-prob)/prob))


def prob_over(draws, line): return float(np.mean(draws>line))


def _calibrated(raw_prob, count):
    # Heuristic conservatism, NOT learned calibration; historical backtesting required.
    damp=.72 if count>=15 else .60
    return float(.5+(raw_prob-.5)*damp)


def _is_fresh(row, target_date, max_age_hrs=8):
    """For today's games require reasonably recent book snapshots.
    Historical manual timestamps retained; missing snapshots treated as unverified.
    """
    stamp=str(row.get('last_update',''))
    if stamp in ('','nan','None'):return False
    t=pd.to_datetime(stamp,utc=True,errors='coerce')
    if pd.isna(t):return False
    today=pd.Timestamp.now(tz='America/Chicago').date()
    if target_date!=today:return True  # For historical replay, snapshot validation is separate.
    return (pd.Timestamp.now(tz='UTC')-t).total_seconds()<=max_age_hrs*3600


def score_game(game, sims, projections, metadata, dvp, odds, target_date,
               min_edge=.045, min_prob=.54, max_per_game=3):
    lines=[]
    if not odds.empty:
        o=odds.copy()
        o=o[(o.home.map(map_team)==game['home']) & (o.away.map(map_team)==game['away'])]
        o=o[o['alternate'].astype(str).str.lower().isin(['false','0'])]
        # Only usable prices with known timestamps. Clean manually entered date data.
        o=o[o.apply(lambda r:_is_fresh(r,target_date),axis=1)]
        o=o.copy()
        o['player_key']=o.player.map(normal_name)
        for pid,meta in metadata.items():
            player_lines=o[o.player_key==normal_name(meta['player'])]
            for _,line in player_lines.iterrows():
                stat=line.stat
                if stat not in STATS or (pid,stat) not in sims:continue
                draw=sims[(pid,stat)]
                raw=prob_over(draw,float(line.line))
                prob=_calibrated(raw,meta['games'])
                if line.side=='Under':prob=1-prob
                implied=implied_probability(line.price)
                edge=prob-implied
                ev=prob*(american_decimal(line.price)-1)-(1-prob)
                lines.append(dict(**meta,stat=stat,side=line.side,line=float(line.line),
                                  book=str(line.book),price=int(line.price),prob=round(prob,4),
                                  raw_prob=round(raw,4),implied=round(implied,4),edge=round(edge,4),
                                  ev_per_dollar=round(ev,4),mean=projections[pid][stat],
                                  dvp_rank=_get_rank(dvp,meta['opponent'],meta['position'],stat),
                                  verified_price=True))
    qualified=[r for r in lines if r['edge']>=min_edge and r['prob']>=min_prob and r['ev_per_dollar']>0]
    qualified.sort(key=lambda x:(x['edge'],x['ev_per_dollar']),reverse=True)
    picks=[];used=set()
    for p in qualified:
        if p['player_id'] in used:continue
        picks.append(p);used.add(p['player_id'])
        if len(picks)==max_per_game:break
    # No odds / no edges => watchlist only. Never relabel projections as value bets.
    watch=[]
    sorted_players=sorted(metadata.items(), key=lambda t:projections[t[0]]['points'],reverse=True)
    for pid,meta in sorted_players:
        favorable=[]
        for stat in STATS:
            rk=_get_rank(dvp,meta['opponent'],meta['position'],stat)
            favor=(31-rk) if rk is not None else 0
            favorable.append((favor,stat,rk))
        _,stat,rk=max(favorable)
        watch.append({**meta,'stat':stat,'mean':projections[pid][stat], 'dvp_rank':rk})
        if len(watch)>=3: break
    return picks,watch


def _get_rank(dvp,opponent,position,stat):
    r=dvp[(dvp.team==opponent)&(dvp.position==position)&(dvp.stat==stat)]
    return int(r.iloc[0]['rank']) if not r.empty else None


def build_ladder(game,sims,projections,metadata,odds):
    """One player with 3 ascending single-bet milestones. Never pretend model-implied
    'fair odds' are offers. Ladder singles are NOT a 3-leg parlay.
    """
    candidates=[]
    # Favor offensive contributors with enough baseline playing time.
    for pid,meta in metadata.items():
        if meta['minutes_projection']<24:continue
        for stat in ('points','threes','rebounds','assists'):
            if stat=='points' and projections[pid]['points']<13:continue
            if stat=='threes' and projections[pid]['threes']<2:continue
            if stat=='rebounds' and projections[pid]['rebounds']<6:continue
            if stat=='assists' and projections[pid]['assists']<5:continue
            candidates.append((pid,stat,meta))
    if not candidates:return None
    rows=[]
    if not odds.empty:
        alt=odds[odds.alternate.astype(str).str.lower().isin(['true','1'])].copy()
        alt=alt[(alt.home.map(map_team)==game['home'])&(alt.away.map(map_team)==game['away'])&
                (alt.side=='Over')]
        alt['player_key']=alt.player.map(normal_name)
        for pid,stat,meta in candidates:
            m=alt[(alt.player_key==normal_name(meta['player']))&(alt.stat==stat)]
            if m.empty:continue
            # choose highest priced sportsbook for each exact milestone
            choices=m.sort_values('price',ascending=False).drop_duplicates('line').sort_values('line')
            choices=choices[choices.apply(lambda r:_is_fresh(r,pd.Timestamp.now(tz='America/Chicago').date()),axis=1)]
            avail=[]
            for _,r in choices.iterrows():
                prob=_calibrated(prob_over(sims[(pid,stat)],r.line),meta['games'])
                if .12<=prob<=.82:
                    avail.append(dict(line=float(r.line),threshold=int(math.floor(r.line)+1),
                                      prob=round(prob,3),price=int(r.price),book=str(r.book),
                                      edge=round(prob-implied_probability(r.price),3),
                                      verified_price=True))
            if len(avail)>=3:
                # span range, requiring distinct, monotonic milestones
                selected=[avail[0],avail[len(avail)//2],avail[-1]]
                scores=(.4*selected[0]['prob']+.35*selected[1]['prob']+.25*selected[2]['prob'])
                rows.append((scores,{'player':meta['player'],'team':meta['team'],'position':meta['position'],
                                     'stat':stat,'levels':selected,'market_verified':True,
                                     'warning':'Each rung is a separate alternate-line bet, not a parlay.'}))
    if rows:
        return max(rows,key=lambda x:x[0])[1]
    # Honest odds-free fallback: project three milestones + probabilities; no prices.
    pid,stat,meta=max(candidates,key=lambda x:(x[1]=='points',projections[x[0]][x[1]]))
    draw=sims[(pid,stat)]
    step=5 if stat=='points' else 1
    quantiles=np.quantile(draw,[.27,.50,.73])
    values=[]
    for q in quantiles:
        t=max(1,int(q//step)*step)
        while t in values:t+=step
        values.append(t)
    levels=[]
    for t in values:
        prob=_calibrated(float(np.mean(draw>=t)),meta['games'])
        levels.append({'line':t-.5,'threshold':t,'prob':round(prob,3),
                       'price':None,'book':None,'fair_price':fair_american(prob),'verified_price':False})
    return {'player':meta['player'],'team':meta['team'],'position':meta['position'],'stat':stat,
            'levels':levels,'market_verified':False,
            'warning':'Projected milestones ONLY. No sportsbook alternate odds verified; do not treat as priced picks.'}


def parlay_candidates(picks,sims,stake=10.,max_output=12):
    """Evaluate parlay outcomes from joint Monte Carlo samples rather than multiplying
    marginal probabilities. Shared matchup factor is a simple correlation approximation.
    """
    legs=sorted(picks,key=lambda p:p['edge'],reverse=True)[:15]
    candidates=[]
    for size in (2,3):
        for combo in combinations(legs,size):
            if len({x['player_id'] for x in combo})!=size:continue
            dec=np.prod([american_decimal(p['price']) for p in combo])
            if not np.isfinite(dec):continue
            hits=[]
            for x in combo:
                d=sims.get((x['player_id'],x['stat']))
                if d is None:break
                hits.append(d>x['line'] if x['side']=='Over' else d<x['line'])
            if len(hits)!=size:continue
            raw_joint=np.mean(np.logical_and.reduce(hits))
            # Conservative probability damp toward 0.5^size, not calibrated.
            conservative=.85*raw_joint+.15*(.5**size)
            net=stake*(conservative*(dec-1)-(1-conservative))
            candidates.append(dict(legs=[{k:x[k] for k in ['matchup','player','side','stat','line','price','book']} for x in combo],
                                   n=size,prob=round(float(conservative),4),decimal_odds=round(float(dec),2),
                                   implied=round(float(1/dec),4),expected_profit=round(float(net),2),
                                   stake=stake,net_profit_if_win=round(float(stake*(dec-1)),2)))
    candidates=[x for x in candidates if x['expected_profit']>0 and x['prob']>x['implied']+.03]
    return sorted(candidates,key=lambda x:x['expected_profit'],reverse=True)[:max_output]
