import json, glob, os, numpy as np
rows=[]
for f in sorted(glob.glob("/workspace/gen/*.json")):
    inst=os.path.basename(f)[:-5]
    try:
        d=json.load(open(f)); fr=d["frames"][1:-1]
    except Exception as e:
        print("skip", inst, e); continue
    if len(fr) < 50: print("short", inst, len(fr)); continue
    pen=np.array([x["pen_mm"] for x in fr]); gt=np.array([x["grip_touch_x"] for x in fr]); err=np.array([x["pos_err_m"] for x in fr])*100; rot=np.degrees([x["rot_err_rad"] for x in fr])
    log=open(f"/workspace/gen/{inst}.log").read(); succ=log.split("tot_obj_succ_nn:")[-1].split()[0] if "tot_obj_succ_nn:" in log else "?"
    held = float(err[-1]) < 10.0
    rows.append(dict(inst=inst, succ=succ, err=float(err.mean()), err_end=float(err[-1]), rot=float(rot.mean()), pen=float(pen.mean()), pen_med=float(np.median(pen)), pen_max=float(pen.max()), f2=float((pen>2).mean()), f5=float((pen>5).mean()), grip_med=float(np.median(gt)), grip_max=float(gt.max()), f40=float((gt>40).mean()), held=held, w=d["summary"]["object_weight_n"], touch=float(np.mean([x["n_links"]>0 for x in fr]))))
print(f"{'clip':36s} succ  err_cm  end_cm  rot  pen_mean med max  >2mm >5mm  grip_med max >40x  touch held")
for r in sorted(rows, key=lambda r: r["err"]):
    print(f"{r['inst']:36s} {r['succ']:>4s}  {r['err']:6.2f} {r['err_end']:7.1f} {r['rot']:4.0f}  {r['pen']:5.2f} {r['pen_med']:4.2f} {r['pen_max']:5.1f}  {r['f2']*100:3.0f}% {r['f5']*100:3.0f}%  {r['grip_med']:5.0f} {r['grip_max']:5.0f} {r['f40']*100:3.0f}%  {r['touch']*100:3.0f}% {int(r['held'])}")
n=len(rows); held=[r for r in rows if r["held"]]; tr=[r for r in rows if r["err"]<10]
print(f"\nn={n}  held at end (<10 cm): {len(held)}  mean err<10cm: {len(tr)}  their success counter >0: {sum(1 for r in rows if r['succ'] not in ('0','?'))}")
if held:
    pen=np.array([r["pen"] for r in held]); f2=np.array([r["f2"] for r in held]); f5=np.array([r["f5"] for r in held]); gm=np.array([r["grip_med"] for r in held])
    print(f"over held clips: pen mean median {np.median(pen):.2f} mm (range {pen.min():.2f}-{pen.max():.2f}); frames>2mm median {np.median(f2)*100:.0f}% (range {f2.min()*100:.0f}-{f2.max()*100:.0f}); >5mm median {np.median(f5)*100:.0f}%; grip median of medians {np.median(gm):.0f}x")
json.dump(rows, open("/workspace/gen/summary.json","w"), indent=1)
