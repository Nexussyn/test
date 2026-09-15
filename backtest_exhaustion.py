#!/usr/bin/env python3
import argparse,time
from datetime import datetime,timezone
import numpy as np,pandas as pd,requests
from scipy import stats
BASE='https://fapi.binance.com'; BAR=pd.Timedelta(minutes=5); MS=300000; SYM='BTCUSDT'
def api(s,path,params):
    for a in range(6):
        try:
            r=s.get(BASE+path,params=params,timeout=30)
            if r.status_code in (418,429,500,502,503,504): time.sleep(min(8,.5*2**a)); continue
            r.raise_for_status(); return r.json()
        except requests.RequestException:
            if a==5: raise
            time.sleep(min(8,.5*2**a))
    raise RuntimeError('API failure')
def klines(s,start,end):
    rows=[]; cur=start
    while cur<end:
        b=api(s,'/fapi/v1/klines',dict(symbol=SYM,interval='5m',startTime=cur,endTime=end,limit=1500))
        if not b: break
        rows+=b; nxt=int(b[-1][0])+MS
        if nxt<=cur: raise RuntimeError('kline cursor stalled')
        cur=nxt; time.sleep(.05)
    c=['ts','open','high','low','close','volume','close_ts','quote','trades','taker_buy','taker_buy_quote','ignore']
    d=pd.DataFrame(rows,columns=c)
    for x in ['open','high','low','close','volume','taker_buy']: d[x]=pd.to_numeric(d[x])
    d['ts']=pd.to_datetime(d.ts,unit='ms',utc=True); d=d.drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
    d['delta']=2*d.taker_buy-d.volume; d['flow']=d.delta/d.volume.replace(0,np.nan); return d
def oi_hist(s,start,end):
    rows=[]; cur=start
    while cur<end:
        b=api(s,'/futures/data/openInterestHist',dict(symbol=SYM,period='5m',startTime=cur,endTime=end,limit=500))
        if not b: break
        rows+=b; nxt=int(b[-1]['timestamp'])+1
        if nxt<=cur: raise RuntimeError('OI cursor stalled')
        cur=nxt; time.sleep(.05)
    d=pd.DataFrame(rows)
    if d.empty: raise RuntimeError('No OI data')
    d['ts']=pd.to_datetime(d.timestamp,unit='ms',utc=True); d['oi']=pd.to_numeric(d.sumOpenInterest)
    return d[['ts','oi']].drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
def prepare(k,o):
    a=k.merge(o.rename(columns={'oi':'oi0'}),on='ts',how='left'); e=o.copy(); e.ts=e.ts-BAR; e=e.rename(columns={'oi':'oi1'})
    a=a.merge(e,on='ts',how='left'); a['doi']=a.oi1-a.oi0; a['doi_pct']=a.doi/a.oi0; a['entry']=a.open.shift(-2); a['future_ok']=True
    for i in range(1,14): a['future_ok'] &= (a.ts.shift(-i)-a.ts.shift(-(i-1))==BAR)
    a['pers']=a.delta.shift(-1).abs()/a.delta.abs().replace(0,np.nan); a['price_resp']=(a.close-a.open).abs()/a.open; a['elas']=a.price_resp/a.delta.abs().replace(0,np.nan); a['elas_decay']=1-a.elas.shift(-1)/a.elas.replace(0,np.nan)
    for m,sh in {5:2,15:4,30:7,60:13}.items():
        a[f'cl{m}']=a.close.shift(-sh); a[f'rl{m}']=a[f'cl{m}']/a.entry-1; a[f'rs{m}']=-a[f'rl{m}']
        lo=pd.concat([a.low.shift(-i) for i in range(2,sh+1)],axis=1).min(axis=1); hi=pd.concat([a.high.shift(-i) for i in range(2,sh+1)],axis=1).max(axis=1)
        a[f'maeL{m}']=lo/a.entry-1; a[f'mfeL{m}']=hi/a.entry-1; a[f'maeS{m}']=(a.entry-hi)/a.entry; a[f'mfeS{m}']=(a.entry-lo)/a.entry
    return a
def masks(d,thr=.25,vol=500,pmin=.5,emin=.5):
    buy=(d.flow>=thr)&(d.volume>=vol); sell=(d.flow<=-thr)&(d.volume>=vol); d['side']=np.select([buy,sell],['BUY','SELL'],default='NONE'); shock=d.side!='NONE'; oi=d.doi<0; decay=d.pers<1
    fail=((d.side=='BUY')&(d.high.shift(-1)<=d.high)&(d.close.shift(-1)<d.close))|((d.side=='SELL')&(d.low.shift(-1)>=d.low)&(d.close.shift(-1)>d.close)); p=d.pers>=pmin; e=d.elas_decay>=emin
    return {'flow':shock&d.future_ok,'flow_oi':shock&oi&d.future_ok,'flow_oi_decay':shock&oi&decay&d.future_ok,'full':shock&oi&decay&fail&d.future_ok,'strict':shock&oi&decay&fail&p&e&d.future_ok}
def returns(e):
    e=e.copy(); buy=e.side=='BUY'
    for m in [5,15,30,60]: e[f'r{m}']=np.where(buy,e[f'rs{m}'],e[f'rl{m}']); e[f'mae{m}']=np.where(buy,e[f'maeS{m}'],e[f'maeL{m}']); e[f'mfe{m}']=np.where(buy,e[f'mfeS{m}'],e[f'mfeL{m}'])
    return e
def decluster(e,mins=30):
    if e.empty:return e.copy()
    e=e.sort_values('ts').reset_index(drop=True); out=[]; gap=pd.Timedelta(minutes=mins)
    for _,r in e.iterrows():
        if not out or r.ts-out[-1].ts>=gap: out.append(r)
        elif abs(r.flow)>abs(out[-1].flow): out[-1]=r
    return pd.DataFrame(out).reset_index(drop=True)
def boot(v,B=10000):
    v=np.asarray(pd.Series(v).dropna(),float)
    if len(v)<2:return (np.nan,np.nan)
    rng=np.random.default_rng(42); x=v[rng.integers(0,len(v),(B,len(v)))].mean(1); return tuple(np.quantile(x,[.025,.975]))
def summarize(e,name,fee=.0004,slip=.0002):
    if e.empty:return {'set':name,'N':0}
    r=e.r15.dropna(); net=r-fee-slip; w=r[r>0]; l=-r[r<0]; t,p=stats.ttest_1samp(r,0) if len(r)>1 else (np.nan,np.nan); sp=stats.binomtest((r>0).sum(),len(r),p=.5,alternative='greater').pvalue if len(r) else np.nan; lo,hi=boot(r)
    return {'set':name,'N':len(r),'N_BUY':int((e.side=='BUY').sum()),'N_SELL':int((e.side=='SELL').sum()),'mean15':r.mean(),'median15':r.median(),'wr15':(r>0).mean(),'mean30':e.r30.mean(),'mean60':e.r60.mean(),'mean15_net':net.mean(),'PF':w.sum()/l.sum() if l.sum()>0 else np.inf,'PF_net':net[net>0].sum()/(-net[net<0]).sum() if (-net[net<0]).sum()>0 else np.inf,'t':t,'p_t':p,'p_sign':sp,'boot_low':lo,'boot_high':hi,'MAE30':e.mae30.mean(),'MFE30':e.mfe30.mean()}
def rand_test(pool,actual,B=10000):
    r=actual.r15.dropna().to_numpy(float); nb=int((actual.side=='BUY').sum()); ns=int((actual.side=='SELL').sum())
    if not len(r):return {'N':0,'actual_mean':np.nan,'null_mean':np.nan,'p':np.nan}
    pb=pool[pool.side=='BUY'].rs15.dropna().to_numpy(float); ps=pool[pool.side=='SELL'].rl15.dropna().to_numpy(float); rng=np.random.default_rng(123); null=[]
    for _ in range(B):
        z=[]
        if nb:z.append(rng.choice(pb,nb,replace=False))
        if ns:z.append(rng.choice(ps,ns,replace=False))
        null.append(np.concatenate(z).mean())
    null=np.asarray(null); return {'N':len(r),'actual_mean':r.mean(),'null_mean':null.mean(),'null_p975':np.quantile(null,.975),'p':(1+(null>=r.mean()).sum())/(B+1)}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--days',type=int,default=30); ap.add_argument('--flow-threshold',type=float,default=.25); ap.add_argument('--min-volume',type=float,default=500); ap.add_argument('--persistence-min',type=float,default=.5); ap.add_argument('--elasticity-decay-min',type=float,default=.5); ap.add_argument('--cooldown',type=int,default=30); args=ap.parse_args()
    now=int(time.time()*1000); n5=now//MS*MS; start=n5-args.days*86400000; end=n5+MS; s=requests.Session(); s.headers['User-Agent']='deep-btc-exhaustion/1.0'
    print('DOWNLOAD',datetime.fromtimestamp(start/1000,timezone.utc),datetime.fromtimestamp(end/1000,timezone.utc)); k=klines(s,start,end); o=oi_hist(s,start-MS,end+MS); d=prepare(k,o); raw_d=d.copy()
    d=d.dropna(subset=['doi','entry','cl60']).copy(); cutoff=d.ts.max()-pd.Timedelta(minutes=65); d=d[d.ts<=cutoff].reset_index(drop=True)
    pd.DataFrame([{'kline_rows':len(k),'oi_rows':len(o),'kline_gaps':int((k.ts.diff().dropna()!=BAR).sum()),'oi_gaps':int((o.ts.diff().dropna()!=BAR).sum()),'rows_no_oi_raw':int(raw_d.doi.isna().sum()),'analysis_rows':len(d),'cutoff':str(cutoff)}]).to_csv('data_quality.csv',index=False)
    M=masks(d,args.flow_threshold,args.min_volume,args.persistence_min,args.elasticity_decay_min); rows=[]
    for name,m in M.items(): raw=returns(d.loc[m].copy()); raw.to_csv(f'events_{name}_raw.csv',index=False); e=decluster(raw,args.cooldown); e.to_csv(f'events_{name}.csv',index=False); rows.append(summarize(e,name))
    pd.DataFrame(rows).to_csv('backtest_summary.csv',index=False); strict=decluster(returns(d.loc[M['strict']].copy()),args.cooldown)
    pd.DataFrame([summarize(strict[strict.side==x],f'strict_{x}') for x in ['BUY','SELL']]).to_csv('direction_split.csv',index=False)
    costs=[]
    for bps in [0,4,6,10,20,30,50]:
        n=strict.r15.dropna()-bps/10000; costs.append({'cost_bps_rt':bps,'N':len(n),'mean_net':n.mean(),'median_net':n.median(),'wr_net':(n>0).mean(),'PF_net':n[n>0].sum()/(-n[n<0]).sum() if (-n[n<0]).sum()>0 else np.inf})
    pd.DataFrame(costs).to_csv('cost_sensitivity.csv',index=False); pool=returns(d.loc[M['flow']].copy()); pd.DataFrame([rand_test(pool,strict)]).to_csv('matched_randomization_15m.csv',index=False)
    split=int(len(d)*.7); oo=d.iloc[split:].copy(); om=masks(oo,args.flow_threshold,args.min_volume,args.persistence_min,args.elasticity_decay_min); oe=decluster(returns(oo.loc[om['strict']].copy()),args.cooldown); pd.DataFrame([summarize(oe,'strict_OOS')]).to_csv('out_of_sample_summary.csv',index=False)
    pd.DataFrame([{'N':len(d),'mean_long15':d.rl15.mean(),'median_long15':d.rl15.median(),'wr_long15':(d.rl15>0).mean(),'boot_low':boot(d.rl15)[0],'boot_high':boot(d.rl15)[1]}]).to_csv('baseline_15m.csv',index=False)
    grid=[]
    for th in [.20,.25,.30,.35,.40]:
      for v in [250,500,1000]:
       for pmin in [.25,.50,.75]:
        for emin in [.25,.50,.75]:
         mm=masks(d,th,v,pmin,emin)['strict']; ee=decluster(returns(d.loc[mm].copy()),args.cooldown); r=ee.r15.dropna(); pv=stats.ttest_1samp(r,0).pvalue if len(r)>1 else np.nan; grid.append({'flow_threshold':th,'min_volume':v,'persistence_min':pmin,'elasticity_decay_min':emin,'N':len(r),'mean15':r.mean() if len(r) else np.nan,'wr15':(r>0).mean() if len(r) else np.nan,'pvalue':pv})
    g=pd.DataFrame(grid); p=g.pvalue.to_numpy(float); valid=np.isfinite(p); q=np.full(len(g),np.nan); vals=p[valid]; order=np.argsort(vals); adj=np.minimum.accumulate((vals[order]*len(vals)/np.arange(1,len(vals)+1))[::-1])[::-1]; tmp=np.empty(len(vals)); tmp[order]=np.minimum(adj,1); q[valid]=tmp; g['BH_q_exploratory']=q; g.to_csv('threshold_grid.csv',index=False)
    with open('README_results.txt','w') as f:
        f.write('DEEP BTCUSDT 5m EXHAUSTION BACKTEST\n'); f.write(f'Generated UTC: {datetime.now(timezone.utc).isoformat()}\n'); f.write('Frozen: |Delta|/Volume>=25%, volume>=500, OI destruction, next |Delta| lower but >=50%, next-bar price failure, elasticity decay>=50%, entry=t+2 open.\n'); f.write('OI alignment: OI timestamp is period END, so doi=OI[t+5m]-OI[t].\n\n'); f.write(pd.DataFrame([summarize(strict,'STRICT')]).to_string(index=False)); f.write('\n\nMATCHED RANDOMIZATION\n'+pd.read_csv('matched_randomization_15m.csv').to_string(index=False)); f.write('\n\nOOS\n'+pd.read_csv('out_of_sample_summary.csv').to_string(index=False)); f.write('\n\nExploratory threshold grid is not confirmatory and must not be used to re-optimize the frozen rule after observing results.\n')
    print(pd.read_csv('backtest_summary.csv').to_string(index=False)); print('\nSTRICT',summarize(strict,'strict')); print('\nRANDOM',rand_test(pool,strict)); print('\nOOS',pd.read_csv('out_of_sample_summary.csv').to_string(index=False))
if __name__=='__main__': main()
