from __future__ import annotations
import os, json, math, shutil, hashlib
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO = "ibrahimdaud/binance-btcusdt"
RAW_REPO = "delmiron27/cryptolake-binance-futures-btc"
OUT = Path("output")
CACHE = Path("cache")
OUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
START = pd.Timestamp(os.getenv("START_DATE","2025-01-01"), tz="UTC")
END = pd.Timestamp(os.getenv("END_DATE","2026-05-31"), tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
MODE = os.getenv("DATA_SOURCE","hf_features")

FEATURE_COLUMNS = [
    "bar_time_ms","symbol","close","log_ret_1m","log_ret_5m","log_ret_15m","log_ret_60m",
    "realized_vol_30m","rsi_14","vol_5m","taker_buy_ratio_5m","trade_count_5m","avg_trade_size_5m",
    "depth_imbalance_1pct","vpin_50","vpin_bucket_imbalance","hawkes_buy_intensity",
    "hawkes_sell_intensity","hawkes_net","oi_btc","oi_change_1h","ls_count_ratio",
    "taker_ls_vol_ratio","fwd_ret_5m","fwd_ret_15m","fwd_ret_60m","fwd_direction_5m"
]

def log(x): print(f"[{datetime.now(timezone.utc).isoformat()}] {x}", flush=True)

def dt_from_any(v):
    if pd.api.types.is_integer_dtype(pd.Series([v])):
        # Heuristic based on magnitude.
        unit = "ns" if abs(int(v)) > 10**17 else "ms"
        return pd.to_datetime(v, unit=unit, utc=True)
    return pd.to_datetime(v, utc=True)

def list_files(repo, patterns=None):
    api = HfApi()
    out=[]
    for x in api.list_repo_tree(repo_id=repo, repo_type="dataset", recursive=True):
        path = getattr(x, "path", "")
        if not path or not path.endswith(".parquet"): continue
        if patterns and not any(p.lower() in path.lower() for p in patterns): continue
        out.append(path)
    return sorted(out)

def get_features():
    log("Discovering 5-minute feature partitions")
    paths = list_files(REPO, ["features/BTCUSDT/"])
    frames=[]
    for p in paths:
        # date is encoded in the filename in the source dataset.
        stem = Path(p).stem
        try: day = pd.Timestamp(stem, tz="UTC")
        except Exception: continue
        if day < START.normalize() or day > END.normalize(): continue
        local = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=p, local_dir=str(CACHE/"features"))
        df = pd.read_parquet(local, columns=[c for c in FEATURE_COLUMNS if c in pd.read_parquet(local, engine="pyarrow", columns=[]).schema.names])
        frames.append(df)
        log(f"feature {day.date()} rows={len(df)}")
    if not frames: raise RuntimeError("No feature partitions found")
    df=pd.concat(frames, ignore_index=True)
    df["ts"]=pd.to_datetime(df["bar_time_ms"], unit="ms", utc=True)
    df=df[(df.ts>=START)&(df.ts<=END)].sort_values("ts").drop_duplicates("ts")
    return df

def choose_stream_paths():
    # Discover streams instead of hard-coding the exact producer names.
    allp=list_files(RAW_REPO)
    groups={"book":[],"liq":[],"funding":[],"oi":[],"trade":[]}
    for p in allp:
        q=p.lower()
        if "/book/" in q or "/orderbook/" in q: groups["book"].append(p)
        elif "forceorder" in q or "liquidat" in q: groups["liq"].append(p)
        elif "/funding/" in q or "mark" in q: groups["funding"].append(p)
        elif "open_interest" in q or "/oi/" in q or "/openinterest/" in q: groups["oi"].append(p)
        elif "/trade/" in q or "/trades/" in q: groups["trade"].append(p)
    return groups

def read_partition(repo,path):
    local=hf_hub_download(repo_id=repo, repo_type="dataset", filename=path, local_dir=str(CACHE/"raw"))
    try:
        return pd.read_parquet(local)
    finally:
        # Remove downloaded daily file after processing to protect runner disk.
        try: Path(local).unlink(missing_ok=True)
        except Exception: pass

def aggregate_book(paths):
    outs=[]
    for p in paths:
        daypart=p.split("dt=")[-1].split("/")[0] if "dt=" in p else ""
        try:
            day=pd.Timestamp(daypart,tz="UTC")
            if day < START.normalize() or day > END.normalize(): continue
        except Exception: pass
        df=read_partition(RAW_REPO,p)
        if df.empty: continue
        cols={c.lower():c for c in df.columns}
        t=cols.get("timestamp") or cols.get("event_time")
        if not t: continue
        df["ts"]=pd.to_datetime(df[t], unit="ns" if pd.to_numeric(df[t],errors="coerce").dropna().abs().median()>1e17 else "ms", utc=True).dt.floor("5min")
        bids=[cols.get(f"bid_{i}_size") for i in range(20) if cols.get(f"bid_{i}_size")]
        asks=[cols.get(f"ask_{i}_size") for i in range(20) if cols.get(f"ask_{i}_size")]
        if not bids or not asks: continue
        df["bid_depth"]=df[bids].sum(axis=1,numeric_only=True)
        df["ask_depth"]=df[asks].sum(axis=1,numeric_only=True)
        df["depth_imbalance_l2"]=(df.bid_depth-df.ask_depth)/(df.bid_depth+df.ask_depth).replace(0,pd.NA)
        x=df.groupby("ts").agg(depth_imbalance_l2=("depth_imbalance_l2","last"), depth_bid=("bid_depth","last"), depth_ask=("ask_depth","last")).reset_index()
        outs.append(x)
    return pd.concat(outs,ignore_index=True).drop_duplicates("ts") if outs else pd.DataFrame()

def aggregate_liq(paths):
    outs=[]
    for p in paths:
        df=read_partition(RAW_REPO,p)
        if df.empty: continue
        c={c.lower():c for c in df.columns}
        t=c.get("timestamp") or c.get("event_time") or c.get("time")
        if not t: continue
        tv=pd.to_numeric(df[t],errors="coerce")
        df["ts"]=pd.to_datetime(tv,unit="ns" if tv.dropna().abs().median()>1e17 else "ms",utc=True).dt.floor("5min")
        qty=c.get("executed_qty") or c.get("orig_qty") or c.get("quantity") or c.get("qty")
        price=c.get("average_price") or c.get("price") or c.get("avg_price")
        side=c.get("side") or c.get("S")
        df["notional"]=pd.to_numeric(df[qty],errors="coerce")*pd.to_numeric(df[price],errors="coerce") if qty and price else 0.0
        df["liq_count"]=1
        if side:
            sv=df[side].astype(str).str.upper()
            df["liq_long_side"]=(sv=="SELL").astype(int)
            df["liq_short_side"]=(sv=="BUY").astype(int)
        else:
            df["liq_long_side"]=0; df["liq_short_side"]=0
        outs.append(df.groupby("ts").agg(liq_notional=("notional","sum"),liq_count=("liq_count","sum"),liq_long=("liq_long_side","sum"),liq_short=("liq_short_side","sum")).reset_index())
    return pd.concat(outs,ignore_index=True).groupby("ts",as_index=False).sum(numeric_only=True) if outs else pd.DataFrame()

def merge_micro(df):
    groups=choose_stream_paths()
    audit={k:len(v) for k,v in groups.items()}
    log(f"Discovered raw streams: {audit}")
    if groups["book"]:
        b=aggregate_book(groups["book"]); df=df.merge(b,on="ts",how="left")
    else: df["depth_imbalance_l2"]=pd.NA
    if groups["liq"]:
        q=aggregate_liq(groups["liq"]); df=df.merge(q,on="ts",how="left")
    else:
        for c in ["liq_notional","liq_count","liq_long","liq_short"]: df[c]=pd.NA
    # funding/OI/trades are already represented in the feature dataset; retain raw stream counts in provenance.
    df["liq_notional_z_24h"]=(df["liq_notional"]-df["liq_notional"].rolling(288).mean())/df["liq_notional"].rolling(288).std().replace(0,pd.NA)
    df["liq_oi_ratio"]=df["liq_notional"]/df["oi_btc"].replace(0,pd.NA) if "oi_btc" in df else pd.NA
    return df,audit

def main():
    df=get_features()
    audit={"mode":MODE}
    if MODE.lower() in ("microstructure","full","all"):
        df,audit2=merge_micro(df); audit["raw_stream_file_counts"]=audit2
    else:
        # We still expose the richer source capabilities but do not download 44GB by default.
        for c in ["depth_imbalance_l2","liq_notional","liq_count","liq_long","liq_short","liq_notional_z_24h","liq_oi_ratio"]: df[c]=pd.NA
    # Scientific safety: mark future-return columns explicitly; strategy must never use them as features.
    future=[c for c in df.columns if c.startswith("fwd_")]
    df.attrs={"future_columns":future}
    out=OUT/"btcusdt_microstructure_5m.parquet"
    df.to_parquet(out,index=False,compression="zstd")
    quality=pd.DataFrame([{
        "rows":len(df),"start":str(df.ts.min()),"end":str(df.ts.max()),
        "duplicate_ts":int(df.ts.duplicated().sum()),"null_close":int(df.close.isna().sum()),
        "depth_available_pct":float(df.depth_imbalance_l2.notna().mean()) if "depth_imbalance_l2" in df else 0,
        "liquidation_available_pct":float(df.liq_notional.notna().mean()) if "liq_notional" in df else 0
    }])
    quality.to_csv(OUT/"quality_report.csv",index=False)
    (OUT/"future_columns.json").write_text(json.dumps(future,indent=2))
    manifest={"generated_at":datetime.now(timezone.utc).isoformat(),"repo":REPO,"raw_repo":RAW_REPO,"start":str(START),"end":str(END),"mode":MODE,"canonical":str(out),"quality":quality.iloc[0].to_dict(),"audit":audit}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2,default=str))
    log(f"DONE rows={len(df)} canonical={out}")

if __name__=="__main__": main()
