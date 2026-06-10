import json
import yaml
import re
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_reasoning(reasoning):
    metrics = {}
    
    # ADX
    m = re.search(r'ADX[^\d]*(\d+\.\d+|\d+)', reasoning)
    if m:
        metrics['adx'] = float(m.group(1))
        
    # Volume ratio
    m = re.search(r'volume ratio[^\d]*(\d+\.\d+|\d+)', reasoning.lower())
    if not m:
        m = re.search(r'volume[^\d]*(\d+\.\d+|\d+)x', reasoning.lower())
    if m:
        metrics['vol'] = float(m.group(1))
        
    return metrics

def simulate():
    strategy_file = os.path.join(BASE_DIR, "kronos_strategy.yaml")
    trades_file = os.path.join(BASE_DIR, "state", "all_pnl_trades.jsonl")
    
    if not os.path.exists(trades_file):
        print(f"Error: {trades_file} not found. Please run reflection extraction first.")
        return
        
    with open(strategy_file, "r") as f:
        strat = yaml.safe_load(f)
    params = strat.get("params", {})
    
    min_adx = params.get("min_adx_trending", 18)
    min_vol = params.get("min_prefilter_volume_ratio", 0.3)
    min_conf = params.get("min_confidence", 80)
    
    print(f"Simulating strategy version: {strat.get('version')}")
    print(f"Current Params: min_adx={min_adx}, min_vol={min_vol}, min_conf={min_conf}")
    
    all_trades = [json.loads(x) for x in open(trades_file) if x.strip()]
    
    def evaluate(test_adx, test_vol, test_conf):
        accepted_trades = []
        rejected_by_adx = 0
        rejected_by_vol = 0
        rejected_by_conf = 0
        
        for t in all_trades:
            reasoning = t.get("reasoning", "")
            metrics = parse_reasoning(reasoning)
            
            adx = metrics.get('adx')
            vol = metrics.get('vol')
            conf = int(t.get('confidence', 100))
            
            adx_pass = adx is None or adx >= test_adx
            vol_pass = vol is None or vol >= test_vol
            conf_pass = conf >= test_conf
            
            if adx_pass and vol_pass and conf_pass:
                accepted_trades.append(t)
            else:
                if not adx_pass: rejected_by_adx += 1
                if not vol_pass: rejected_by_vol += 1
                if not conf_pass: rejected_by_conf += 1
                
        closed = [t for t in accepted_trades if t.get('pnl') and str(t['pnl']).lower() != 'none']
        pnl = sum(float(t['pnl']) for t in closed) if closed else 0
        win = sum(1 for t in closed if float(t['pnl']) > 0) if closed else 0
        win_rate = round(win/len(closed)*100, 2) if closed else 0
        
        return {
            "total": len(accepted_trades),
            "pnl": round(pnl, 2),
            "win_rate": win_rate,
            "rej_adx": rejected_by_adx,
            "rej_vol": rejected_by_vol,
            "rej_conf": rejected_by_conf
        }
        
    print("\n--- Sensitivity Analysis ---")
    
    print("\n1. Impact of min_confidence (Baseline ADX=18, Vol=0.1):")
    for c in [75, 78, 80, 82, 85]:
        res = evaluate(18, 0.1, c)
        print(f"Conf={c}: Trades={res['total']}, PnL={res['pnl']}, WinRate={res['win_rate']}% (Rejected solely by Conf: {res['rej_conf']})")
        
    print("\n2. Impact of min_adx_trending (Baseline Conf=80, Vol=0.1):")
    for a in [15, 18, 20, 22, 25]:
        res = evaluate(a, 0.1, 80)
        print(f"ADX={a}: Trades={res['total']}, PnL={res['pnl']}, WinRate={res['win_rate']}% (Rejected solely by ADX: {res['rej_adx']})")
        
    print("\n3. Impact of min_prefilter_volume_ratio (Baseline Conf=80, ADX=18):")
    for v in [0.1, 0.3, 0.5, 0.8, 1.0]:
        res = evaluate(18, v, 80)
        print(f"Vol={v}: Trades={res['total']}, PnL={res['pnl']}, WinRate={res['win_rate']}% (Rejected solely by Vol: {res['rej_vol']})")
        
    print("\n4. Combined Current Params vs Baseline (ADX=18, Vol=0.1, Conf=80):")
    baseline = evaluate(18, 0.1, 80)
    current = evaluate(min_adx, min_vol, min_conf)
    print(f"Baseline: Trades={baseline['total']}, PnL={baseline['pnl']}, WinRate={baseline['win_rate']}%")
    print(f"Current : Trades={current['total']}, PnL={current['pnl']}, WinRate={current['win_rate']}%")

if __name__ == "__main__":
    simulate()
