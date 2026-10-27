import datetime as dt
import unittest
import numpy as np
import pandas as pd
from nba_model.sources import parse_boxscore, map_team, normal_name
from nba_model.demo import generate_demo
from nba_model.model import prepare_logs,dvp_ranking,simulate_matchup,prob_over,parlay_candidates,american_decimal,implied_probability,score_game


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.date=dt.date(2026,10,27)
        self.games,self.logs,self.rosters,self.odds=generate_demo(self.date)
        self.hist=prepare_logs(self.logs,self.date,self.rosters)
        self.dvp=dvp_ranking(self.hist)

    def test_no_lookahead(self):
        future=self.logs.iloc[:1].copy()
        future['date']='2026-10-28'
        all_logs=pd.concat([self.logs,future],ignore_index=True)
        self.assertEqual(len(prepare_logs(all_logs,self.date,self.rosters)),len(self.logs))

    def test_dvp_rank1_largest_allowance(self):
        for _,group in self.dvp.groupby(['position','stat']):
            ranked=group.sort_values('rank')
            self.assertTrue((ranked.allowed_per36.diff().fillna(0)<=.0000001).all())

    def test_sims_shared_player(self):
        sims,proj,meta=simulate_matchup(self.games[0],self.hist,self.dvp,self.rosters,n=1500)
        self.assertGreater(len(sims),8)
        self.assertTrue(set(stat for _,stat in sims)=={'points','rebounds','assists','threes'})
        self.assertTrue(all(len(v)==1500 for v in sims.values()))

    def test_betting_math(self):
        self.assertAlmostEqual(implied_probability(-110),110/210)
        self.assertAlmostEqual(american_decimal(125),2.25)
        self.assertAlmostEqual(prob_over(np.array([10,15,18]),14.5),2/3)

    def test_missing_prices_means_no_bets(self):
        sims,proj,meta=simulate_matchup(self.games[0],self.hist,self.dvp,self.rosters,n=1000)
        picks,watch=score_game(self.games[0],sims,proj,meta,self.dvp,self.odds,self.date)
        self.assertEqual(len(picks),0)
        self.assertEqual(len(watch),3)

    def test_no_fake_parlays(self):
        self.assertEqual(parlay_candidates([],{}),[])

    def test_teams_map_and_names(self):
        self.assertEqual(map_team('Los Angeles Lakers'),'LAL')
        self.assertEqual(normal_name('De\'Aaron Fox'),normal_name('DeAaron Fox'))

    def test_parse_espn_boxscore_structure(self):
        game={'game_id':'42','day':'2026-10-22','away':'BOS','home':'NYK'}
        sample={'boxscore':{'players':[{'team':{'abbreviation':'BOS'},'statistics':[{
            'names':['MIN','3PT','REB','AST','PTS'],'athletes':[{'athlete':{'id':'111','displayName':'Example Player','position':{'abbreviation':'PG'}},'stats':['30','2-6','5','8','20']}]}]}]}}
        rows=parse_boxscore(sample,game)
        self.assertEqual(rows[0]['threes'],2)
        self.assertEqual(rows[0]['opponent'],'NYK')
        self.assertEqual(rows[0]['position'],'PG')

if __name__=='__main__':unittest.main()

class BacktestTests(unittest.TestCase):
    def test_backtest_only_history(self):
        from nba_model.demo import generate_demo
        from nba_model.backtest import backtest
        import datetime as dt
        g,l,r,o=generate_demo(dt.date(2026,10,27))
        result=backtest(l,minimum_history_days=18,max_days=2,simulations=120)
        self.assertTrue(set(result.columns).issuperset({'prob','hit','brier'}))
        self.assertTrue(((result.prob>=0)&(result.prob<=1)).all())

class PriceAwareTests(unittest.TestCase):
    def test_priced_props_and_joint_parlays(self):
        import datetime as dt
        from nba_model.demo import generate_demo
        from nba_model.model import prepare_logs,dvp_ranking,simulate_matchup,score_game,parlay_candidates
        day=dt.date(2026,10,27)
        games,logs,rosters,_=generate_demo(day)
        hist=prepare_logs(logs,day,rosters)
        dvp=dvp_ranking(hist)
        game=games[0]
        sims,proj,meta=simulate_matchup(game,hist,dvp,rosters,n=1200)
        now=dt.datetime.now(dt.timezone.utc).isoformat()
        # Extremely generous FICTITIOUS odds: a fixture, not sportsbook information.
        sample=[]
        for pid,info in list(meta.items())[:3]:
            sample.append({'game_date':day.isoformat(),'home':game['home'],'away':game['away'],
                'player':info['player'],'stat':'points','side':'Over','line':2.5,
                'price':200,'book':'FAKE TEST BOOK','alternate':False,'last_update':now})
        picks,_=score_game(game,sims,proj,meta,dvp,pd.DataFrame(sample),day)
        self.assertEqual(len(picks),3)
        parlays=parlay_candidates(picks,sims)
        self.assertTrue(any(p['n']==2 for p in parlays))
        self.assertTrue(any(p['n']==3 for p in parlays))
