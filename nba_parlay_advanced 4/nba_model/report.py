"""Generate a self-contained dark-mode dashboard; no JavaScript dependencies."""
from __future__ import annotations
import datetime as dt
import html
import json
from pathlib import Path
import pandas as pd

STYLES='''
:root{color-scheme:dark;--bg:#0d1320;--panel:#161f30;--panel2:#1d2a3e;--text:#f0f5fa;--muted:#99a9ba;--line:#314055;--blue:#6ac9ff;--green:#6dd6b2;--gold:#fbc470}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,Segoe UI,sans-serif;font-size:14px;line-height:1.5}
main{max-width:1250px;margin:auto;padding:28px 19px 70px}h1{font-size:32px;margin:6px 0}h2{font-size:23px;margin:33px 0 14px}h3{font-size:19px;margin:0 0 12px}p{color:var(--muted)}
.eyebrow{text-transform:uppercase;letter-spacing:.15em;color:var(--blue);font-size:11px;font-weight:800}.lead{max-width:800px;font-size:16px}
.hero{padding:26px;background:linear-gradient(130deg,#1e3450,#142131 65%);border:1px solid var(--line);border-radius:20px}
.pills{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px}.pill{background:#273c55;border:1px solid #355475;border-radius:50px;padding:6px 12px;color:#dcefff}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:16px 0}.kpi{background:var(--panel);border:1px solid var(--line);padding:17px;border-radius:14px}.kpi strong{display:block;font-size:26px}.kpi small{color:var(--muted)}
.games{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px}.card{border:1px solid var(--line);background:var(--panel);padding:20px;border-radius:16px;min-width:0}.game-title{display:flex;justify-content:space-between;gap:12px;align-items:center}.game-title small{color:var(--muted)}
.selection{border-top:1px solid var(--line);padding:13px 0}.selection:first-child{border-top:0}.selection .name{font-weight:700;color:#fff}.selection .desc{color:var(--muted);font-size:13px}.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:7px}.chip{font-size:12px;border-radius:8px;background:#27374c;padding:3px 8px;color:#b8cbe0}.positive{color:var(--green)}.warn{color:var(--gold)}
.ladder{background:var(--panel2);padding:13px;border-radius:12px;margin-top:14px}.rungs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:9px}.rung{background:#102031;border-radius:9px;padding:12px 8px;text-align:center}.rung strong{display:block;font-size:17px}.rung span{display:block;color:var(--muted);font-size:11px}
.scroll{width:100%;overflow:auto;background:var(--panel);border:1px solid var(--line);border-radius:14px}table{border-collapse:collapse;width:100%;min-width:600px}td,th{text-align:left;padding:10px 12px;border-bottom:1px solid #293448}th{position:sticky;top:0;background:#203149;font-size:12px;color:#bcd0df}tr:hover td{background:#1d2b40}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.filters{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}select{background:#1d2d43;color:white;border:1px solid #4b6278;border-radius:9px;padding:10px 13px;font-size:14px}.info{border-left:3px solid var(--gold);padding:10px 14px;background:#232737;border-radius:7px;margin:18px 0;color:#e8d8a9}
footer{margin-top:36px;color:var(--muted);font-size:12px}a{color:#84d7ff}code{color:#fff}.muted{color:var(--muted)}
'''

def E(s):return html.escape(str(s))

def generate_report(destination,game_reports,dvp,picks,parlays,warnings,day,generated):
    dest=Path(destination);dest.parent.mkdir(parents=True,exist_ok=True)
    qualified=sum(len(g['picks']) for g in game_reports)
    has_prices=any(p['verified_price'] for p in picks)
    cards=[]
    for g in game_reports:
        picks_html=[]
        for p in g['picks']:
            edge=p['edge']*100
            picks_html.append(f'''<div class="selection"><div class="name">{E(p['player'])} — {E(p['side'])} {p['line']:g} {E(p['stat'])}</div>
<div class="desc">{E(p['book'])} {p['price']:+d} · Model projected {p['mean']:g} · {E(p['position'])} vs {E(p['opponent'])}</div>
<div class="chips"><span class="chip positive">Model {p['prob']:.1%}</span><span class="chip">Break-even {p['implied']:.1%}</span><span class="chip positive">Estimated edge {edge:+.1f} pp</span><span class="chip">DVP rank {p['dvp_rank'] or '—'}</span></div></div>''')
        if not picks_html:
            picks_html.append('<p class="warn">No verified positive-edge plays at the configured minimums. Matchup watchlist below is NOT a bet recommendation.</p>')
            for w in g['watch']:
                picks_html.append(f'''<div class="selection"><div class="name">{E(w['player'])} — {E(w['stat'])}</div><div class="desc">Projection {w['mean']:.1f} · {E(w['position'])} vs {E(w['opponent'])} · DVP rank {w['dvp_rank'] or '—'} · odds not verified</div></div>''')
        ladder=g.get('ladder')
        ladder_html='No eligible ladder player in available historical data.'
        if ladder:
            rungs=[]
            for rung in ladder['levels']:
                price=('Offer '+str(f"{rung['price']:+d}")) if rung['price'] is not None else 'Fair '+E(rung.get('fair_price','N/A'))
                rungs.append(f'''<div class="rung"><strong>{rung['threshold']}+ {E(ladder['stat'])}</strong><span>Est. {rung['prob']:.0%}</span><span>{price}</span></div>''')
            scope='BOOK PRICES VERIFIED' if ladder['market_verified'] else 'MODEL-ONLY · NO OFFER'
            ladder_html=f'''<div class="name">{E(ladder['player'])} <span class="chip">{scope}</span></div><div class="rungs">{''.join(rungs)}</div><div class="desc" style="margin-top:8px">{E(ladder['warning'])}</div>'''
        cards.append(f'''<article class="card"><div class="game-title"><h3>{E(g['game']['away'])} @ {E(g['game']['home'])}</h3><small>{E(g['game'].get('day',''))}</small></div>
{''.join(picks_html)}<div class="ladder"><div class="eyebrow">One ladder candidate</div>{ladder_html}</div></article>''')
    props_rows=[]
    for p in picks:
        props_rows.append(f'''<tr><td>{E(p['matchup'])}</td><td>{E(p['player'])}</td><td>{E(p['side'])} {p['line']:g} {E(p['stat'])}</td><td>{E(p['book'])}</td><td class="num">{p['price']:+d}</td><td class="num">{p['prob']:.1%}</td><td class="num">{p['edge']*100:+.1f} pp</td></tr>''')
    parlay_html=[]
    for v in parlays:
        legs=' · '.join(f"{p['player']} {p['side']} {p['line']:g} {p['stat']}" for p in v['legs'])
        parlay_html.append(f'''<tr><td>{v['n']}-leg</td><td>{E(legs)}</td><td class="num">{v['prob']:.1%}</td><td class="num">{v['decimal_odds']:.2f}x</td><td class="num">${v['expected_profit']:+.2f}</td></tr>''')
    dvp_json=json.dumps([{'team':str(r['team']),'position':str(r['position']),'stat':str(r['stat']),
                          'allowed':round(float(r['allowed_per36']),2),'rank':int(r['rank']),
                          'games':int(r['games'])} for _,r in dvp.iterrows()],ensure_ascii=True).replace('</','<\\/') if not dvp.empty else '[]'
    warning_html=''.join(f'<div class="info">{E(w)}</div>' for w in warnings[:8])
    content=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NBA Props Lab | {E(day)}</title><style>{STYLES}</style></head>
<body><main><section class="hero"><div class="eyebrow">NBA Props Lab / v1 advanced Python</div><h1>Daily NBA Model</h1><p class="lead">Daily matchup analysis, defense-versus-position rankings, model-driven player-prop screening, correlated parlay simulations, and one alternate-line ladder watchlist per game.</p>
<div class="pills"><span class="pill">Slate {E(day)}</span><span class="pill">Generated {E(generated)}</span><span class="pill">{'Verified sportsbook snapshots' if has_prices else 'No verified sportsbook odds'}</span></div></section>
<div class="kpis"><div class="kpi"><strong>{len(game_reports)}</strong><small>NBA games</small></div><div class="kpi"><strong>{qualified}</strong><small>Qualified value candidates</small></div><div class="kpi"><strong>{len(dvp[['team']].drop_duplicates()) if not dvp.empty else 0}</strong><small>Teams with defensive sample</small></div><div class="kpi"><strong>{len(parlays)}</strong><small>Price-backed parlay combinations</small></div></div>
{warning_html}
<h2>Game-by-game plays and ladders</h2><div class="games">{''.join(cards) if cards else '<p>No scheduled games returned for the date.</p>'}</div>
<h2>Defense vs. position — all available team rankings</h2><p>Rank <strong>1 = most generous</strong> (allows the most of that statistic per 36 minutes to players labeled at this primary position). Ranks use the teams present in the loaded dataset; they are not necessarily 1–30 until all 30 teams have data. Early-season samples are shrunk toward the league mean.</p>
<div class="filters"><label>Position <select id="pos"><option>PG</option><option>SG</option><option>SF</option><option>PF</option><option>C</option></select></label><label>Statistic <select id="stat"><option value="points">Points</option><option value="rebounds">Rebounds</option><option value="assists">Assists</option><option value="threes">3-pointers made</option></select></label></div>
<div class="scroll"><table><thead><tr><th class="num">Rank</th><th>Defense</th><th>Against position</th><th>Stat</th><th class="num">Allowed per 36</th><th class="num">Games</th></tr></thead><tbody id="dvp-data"></tbody></table></div>
<h2>Qualified player props</h2><div class="scroll"><table><thead><tr><th>Game</th><th>Player</th><th>Play</th><th>Sportsbook</th><th class="num">Odds</th><th class="num">Est. chance</th><th class="num">Edge</th></tr></thead><tbody>{''.join(props_rows) if props_rows else '<tr><td colspan="7">No priced props passed all screening thresholds. No plays are forced.</td></tr>'}</tbody></table></div>
<h2>2-leg and 3-leg parlay simulations</h2><p>Only combinations of selected price-backed candidates appear. Monte Carlo attempts to model shared game pace and within-player statistical dependence; it is an approximation, not proof of profitable odds.</p>
<div class="scroll"><table><thead><tr><th>Type</th><th>Legs</th><th class="num">Joint chance</th><th class="num">Combined decimal</th><th class="num">Est. EV at $10</th></tr></thead><tbody>{''.join(parlay_html) if parlay_html else '<tr><td colspan="5">No eligible positive-model-EV parlays can be priced.</td></tr>'}</tbody></table></div>
<h2>Method and limitations</h2><p>Projections are weighted recent player production per minute, projected minutes, a shrunk defense-versus-position opponent adjustment, and 14,000 shared-game Monte Carlo samples. The current model does not have validated predictive calibration, play-by-play positional assignments, guaranteed starting lineups, confirmed injury status, or transaction-aware projected minutes. Primary positions are roster labels, not possessions defended. Only use books and game/odds timestamps confirmed for the chosen date. Ladder milestones are separate potential single bets, not an actual bundled parlay. No bets are placed.</p>
<footer>Educational analytics only · Model estimates can be inaccurate · Never interpret a favorable matchup alone as positive expected value.</footer></main>
<script>const rows={dvp_json};function update(){{const pos=document.getElementById('pos').value,stat=document.getElementById('stat').value;const list=rows.filter(x=>x.position===pos&&x.stat===stat).sort((a,b)=>a.rank-b.rank);document.getElementById('dvp-data').innerHTML=list.length?list.map(r=>`<tr><td class="num">${{r.rank}}</td><td>${{r.team}}</td><td>${{r.position}}</td><td>${{r.stat}}</td><td class="num">${{r.allowed.toFixed(2)}}</td><td class="num">${{r.games}}</td></tr>`).join(''):'<tr><td colspan="6">Insufficient matching data.</td></tr>';}}
document.getElementById('pos').addEventListener('change',update);document.getElementById('stat').addEventListener('change',update);update();</script></body></html>'''
    dest.write_text(content,encoding='utf-8')
    return dest
