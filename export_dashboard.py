"""Build a self-contained replay dashboard (results/dashboard.html) from a REAL GA run.

Run from the repo root (build_env() in make_figures.py must be filled in):
    python export_dashboard.py --seed 1 --out results
Open results/dashboard.html in any browser. All numbers come from the run;
the coverage footprint is the same approximation used in visualize.py.
"""
import argparse
import json
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

CALLSIGNS = ["Falcon-1", "Shaheen-2", "Chai-Runner-3", "Raven-4", "Kite-5"]
COLORS = ["#6fe3b5", "#f4a73b", "#ef8fb0", "#7fb4ff", "#c8a2ff"]


def collect(env, res, rs_hist, hp, seed):
    from uav_ga.visualize import coverage_mask
    h, w = env.priority.shape
    pys, pxs = np.nonzero(env.priority > 0)
    total = float(env.priority.sum())
    snaps = []
    for gen, fleet in res.snapshots:
        cov = coverage_mask(env, fleet)
        covered = np.nonzero(cov[pys, pxs])[0].tolist()
        share = float(env.priority[cov].sum()) / total if total > 0 else 0.0
        snaps.append({
            "gen": int(gen),
            "fitness": float(fleet.fitness),
            "share": share,
            "covered": covered,
            "routes": [[[float(x), float(y)] for x, y in plan.waypoints()] for plan in fleet.plans],
        })
    return {
        "w": int(w), "h": int(h),
        "hazard": np.round(env.hazard, 2).ravel().tolist(),
        "nofly": env.no_fly.astype(int).ravel().tolist(),
        "camps": [[int(x), int(y)] for x, y in zip(pxs, pys)],
        "depot": [float(env.depot[0]), float(env.depot[1])],
        "snaps": snaps,
        "best": [float(v) for v in res.best_history],
        "mean": [float(v) for v in res.mean_history],
        "random": [float(v) for v in rs_hist],
        "seed": int(seed),
        "evals": int(res.evaluations),
        "gens_run": int(res.generations_run),
        "callsigns": CALLSIGNS, "colors": COLORS,
    }


def render(data, path):
    html = TEMPLATE.replace("__DATA__", json.dumps(data))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--generations", type=int, default=150)
    ap.add_argument("--tournament-k", type=int, default=3)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from uav_ga.ga import Hyperparameters, run_ga
    from uav_ga.baseline import run_random_search
    from make_figures import build_env

    env = build_env()
    hp = replace(Hyperparameters(), seed=a.seed, generations=a.generations,
                 tournament_k=a.tournament_k)
    res = run_ga(env, hp, snapshot_every=5, verbose=False)
    rs_hist, _ = run_random_search(env, res.evaluations, n_drones=hp.n_drones,
                                   n_waypoints=hp.n_waypoints, seed=a.seed, chunk=hp.pop_size)
    data = collect(env, res, rs_hist, hp, a.seed)
    out = os.path.join(a.out, "dashboard.html")
    render(data, out)
    print(f"GA best {res.best.fitness:.3f} | random best {max(rs_hist):.3f} | saved {out}")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Operation Last Light: replay</title>
<style>
:root{--bg:#0c1511;--card:#121e18;--line:#223229;--tx:#dbe9e0;--mut:#8aa397;--ga:#6fe3b5;--mean:#f4a73b;--rs:#9aa5a0}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:14px system-ui,sans-serif;padding:16px}
.wrap{max-width:980px;margin:auto;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px}
h1{font-size:18px;margin:0}.sub{color:var(--mut);margin:2px 0 14px;font-size:13px}
.grid{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:16px}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
canvas{width:100%;height:auto;background:#0a120e;border:1px solid var(--line);border-radius:10px;display:block}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:8px}.st{background:#0e1913;border:1px solid var(--line);border-radius:8px;padding:8px}
.st b{display:block;font-size:20px;margin-top:2px}.st span{color:var(--mut);font-size:12px}
.leg{margin:12px 0 0;padding:0;list-style:none}.leg li{display:flex;align-items:center;gap:8px;margin:4px 0}
.sw{width:12px;height:12px;border-radius:3px;display:inline-block}
.ctl{display:flex;gap:10px;align-items:center;margin-top:14px}.ctl input{flex:1}
button{background:#1c3027;color:var(--tx);border:1px solid var(--line);border-radius:8px;padding:6px 14px;cursor:pointer}
.note{color:var(--mut);font-size:12px;margin-top:12px}
</style></head><body><div class="wrap">
<h1>Operation Last Light</h1>
<div class="sub" id="sub"></div>
<div class="grid"><div><canvas id="map" width="600" height="600"></canvas></div>
<div><div class="stats">
<div class="st"><span>Generation</span><b id="sGen">0</b></div>
<div class="st"><span>Best fitness</span><b id="sFit">0</b></div>
<div class="st"><span>Priority covered (approx)</span><b id="sCov">0%</b></div>
<div class="st"><span>Random search best</span><b id="sRs">0</b></div></div>
<ul class="leg" id="leg"></ul>
<ul class="leg"><li><span class="sw" style="background:#58d68d"></span>Camp covered</li>
<li><span class="sw" style="background:#e5534b"></span>Camp not reached</li>
<li><span class="sw" style="background:#ff7a1f;opacity:.6"></span>Horde density (hazard)</li>
<li><span class="sw" style="background:#3a3f4b"></span>No-fly zone</li></ul></div></div>
<canvas id="chart" width="900" height="220" style="margin-top:14px"></canvas>
<div class="ctl"><button id="play">Play</button><input type="range" id="sl" min="0" value="0">
<span id="lbl" style="min-width:70px;text-align:right"></span></div>
<div class="note" id="note"></div></div>
<script>
const D=__DATA__;
const map=document.getElementById('map'),mc=map.getContext('2d');
const ch=document.getElementById('chart'),cc=ch.getContext('2d');
const S=map.width/D.w,sl=document.getElementById('sl');
sl.max=D.snaps.length-1;
const gx=x=>(x+.5)*S,gy=y=>map.height-(y+.5)*S;
const fin=v=>v.toFixed(3);
document.getElementById('sub').textContent=`Real GA run, seed ${D.seed}: ${D.gens_run} generations, ${D.evals} evaluations`;
document.getElementById('leg').innerHTML=D.snaps[0].routes.map((_,i)=>`<li><span class="sw" style="background:${D.colors[i]}"></span>${D.callsigns[i]}</li>`).join('');
document.getElementById('note').textContent='Coverage uses an approximate sensor footprint for drawing; the official coverage number comes from fitness.py. Random search plotted per pop_size evaluations.';
function drawMap(k){
 const s=D.snaps[k];mc.clearRect(0,0,map.width,map.height);
 for(let y=0;y<D.h;y++)for(let x=0;x<D.w;x++){const i=y*D.w+x;
  if(D.nofly[i]){mc.fillStyle='#3a3f4b';mc.fillRect(x*S,map.height-(y+1)*S,S,S);}
  else if(D.hazard[i]>0.05){mc.fillStyle=`rgba(255,122,31,${Math.min(.7,D.hazard[i]*.8)})`;mc.fillRect(x*S,map.height-(y+1)*S,S,S);}}
 const cov=new Set(s.covered);
 D.camps.forEach(([x,y],i)=>{mc.fillStyle=cov.has(i)?'#58d68d':'#e5534b';mc.beginPath();mc.arc(gx(x),gy(y),S*.28,0,7);mc.fill();});
 s.routes.forEach((r,i)=>{mc.strokeStyle=D.colors[i];mc.lineWidth=3;mc.lineJoin='round';mc.beginPath();
  mc.moveTo(gx(D.depot[0]),gy(D.depot[1]));r.forEach(([x,y])=>mc.lineTo(gx(x),gy(y)));mc.lineTo(gx(D.depot[0]),gy(D.depot[1]));mc.stroke();
  r.forEach(([x,y])=>{mc.fillStyle=D.colors[i];mc.strokeStyle='#fff';mc.lineWidth=1.5;mc.beginPath();mc.arc(gx(x),gy(y),5,0,7);mc.fill();mc.stroke();});});
 mc.fillStyle='gold';mc.strokeStyle='#000';mc.lineWidth=1.5;mc.font='26px sans-serif';mc.textAlign='center';mc.textBaseline='middle';
 mc.strokeText('\u2605',gx(D.depot[0]),gy(D.depot[1]));mc.fillText('\u2605',gx(D.depot[0]),gy(D.depot[1]));
}
function drawChart(g){
 const W=ch.width,H=ch.height,p={l:50,r:14,t:12,b:26};cc.clearRect(0,0,W,H);
 const all=[...D.best,...D.mean,...D.random],lo=Math.min(...all),hi=Math.max(...all);
 const n=Math.max(D.best.length,D.mean.length,D.random.length)-1;
 const X=i=>p.l+i/n*(W-p.l-p.r),Y=v=>H-p.b-(v-lo)/(hi-lo||1)*(H-p.t-p.b);
 cc.strokeStyle='#223229';cc.fillStyle='#8aa397';cc.font='12px sans-serif';cc.lineWidth=1;cc.textAlign='right';
 for(let t=0;t<=4;t++){const v=lo+(hi-lo)*t/4;cc.beginPath();cc.moveTo(p.l,Y(v));cc.lineTo(W-p.r,Y(v));cc.stroke();cc.fillText(v.toFixed(2),p.l-6,Y(v)+4);}
 cc.textAlign='center';[0,.25,.5,.75,1].forEach(f=>cc.fillText(Math.round(f*n),X(f*n),H-8));
 const line=(a,c,w,d)=>{cc.strokeStyle=c;cc.lineWidth=w;cc.setLineDash(d||[]);cc.beginPath();a.forEach((v,i)=>i?cc.lineTo(X(i),Y(v)):cc.moveTo(X(i),Y(v)));cc.stroke();cc.setLineDash([]);};
 line(D.random,'#9aa5a0',2,[6,4]);line(D.mean,'#f4a73b',2);line(D.best,'#6fe3b5',2.5);
 cc.strokeStyle='#dbe9e0';cc.setLineDash([3,4]);cc.beginPath();cc.moveTo(X(g),p.t);cc.lineTo(X(g),H-p.b);cc.stroke();cc.setLineDash([]);
 cc.textAlign='left';[['AI Commander best','#6fe3b5'],['Population mean','#f4a73b'],['Random search','#9aa5a0']].forEach(([t,c],i)=>{cc.fillStyle=c;cc.fillText(t,p.l+10+i*150,p.t+10);});
}
function show(k){
 const s=D.snaps[k];drawMap(k);drawChart(s.gen);
 sGen.textContent=s.gen;sFit.textContent=fin(s.fitness);sCov.textContent=Math.round(s.share*100)+'%';
 const r=Math.min(s.gen,D.random.length-1);sRs.textContent=fin(D.random[r]);
 lbl.textContent=`gen ${s.gen}`;sl.value=k;
}
let timer=null;const play=document.getElementById('play');
play.onclick=()=>{if(timer){clearInterval(timer);timer=null;play.textContent='Play';return;}
 if(+sl.value>=sl.max)sl.value=0;play.textContent='Pause';
 timer=setInterval(()=>{const k=+sl.value+1;if(k>sl.max){clearInterval(timer);timer=null;play.textContent='Play';return;}show(k);},350);};
sl.oninput=()=>show(+sl.value);
show(0);
</script></body></html>"""

if __name__ == "__main__":
    main()