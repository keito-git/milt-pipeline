import json,sys,collections as C
import numpy as np
sys.path.insert(0,'.')
from analyze_blind_audit import wilson,kappa,cluster_ci
R=sys.argv[1]
key={r["id"]:r for r in json.load(open("audit_key.json"))["items"]}
A,B=[json.load(open(f"{R}/auditor{i}.json")) for i in (1,2)]
print({k:(v if not isinstance(v,list) else len(v)) for k,v in B.items()})
a,b=[{r["id"]:r for r in x["answers"] if r.get("label")} for x in (A,B)]
U="不明"
base=[i for i in key if not key[i]["is_hidden_repeat"]]
def rep(name,ids):
    la=[a[i]["label"] for i in ids]; lb=[b[i]["label"] for i in ids]
    dec=[i for i in ids if a[i]["label"]!=U and b[i]["label"]!=U]
    print(name,"n",len(ids),"agree",round(np.mean([x==y for x,y in zip(la,lb)]),4),"k",kappa(la,lb),
      "| both decided",len(dec),"agree",round(np.mean([a[i]["label"]==b[i]["label"] for i in dec]),4) if dec else None,
      "k",kappa([a[i]["label"] for i in dec],[b[i]["label"] for i in dec]))
rep("all",base)
for bd in ["[0.0,0.5)","[0.5,0.7)","[0.7,0.9)","[0.9,1.0]"]:
    rep(bd,[i for i in base if key[i]["band"]==bd and key[i]["stratum"] in("S1_all_high","S2_random")])
rep("rand+high",[i for i in base if key[i]["stratum"] in("S1_all_high","S2_random")])
# consensus: both decided and agree; error of auto / gold
print("--consensus (both decided and same label)")
for bd in ["[0.0,0.5)","[0.5,0.7)","[0.7,0.9)","[0.9,1.0]"]:
    ids=[i for i in base if key[i]["band"]==bd and key[i]["stratum"] in("S1_all_high","S2_random")]
    con=[i for i in ids if a[i]["label"]==b[i]["label"]!=U]
    e=[key[i]["auto_label"]!=a[i]["label"] for i in con]; g=[key[i]["gold"]!=a[i]["label"] for i in con]
    print(bd,"n",len(ids),"consensus",len(con),"autoerr",sum(e),round(sum(e)/len(con),4),wilson(sum(e),len(con)),
      "video",cluster_ci(e,[key[i]["src"] or key[i]["crop"] for i in con]),"golderr",sum(g),wilson(sum(g),len(con)))
print("--high band: all disagreements / auto errors")
hi=[i for i in base if key[i]["band"]=="[0.9,1.0]"]
print("unknown A",sum(a[i]["label"]==U for i in hi),"B",sum(b[i]["label"]==U for i in hi),"both",sum(a[i]["label"]==U==b[i]["label"] for i in hi))
for i in hi:
    k=key[i];x,y=a[i]["label"],b[i]["label"]
    if x!=y or x!=k["auto_label"]:
        print(k["auto_label"],"gold",k["gold"],"A",x,"B",y)
print("--rare, both raters")
for n,f in [("gold_rare",lambda k:k["gold_rare"]),("auto_rare",lambda k:k["auto_rare"])]:
    ids=[i for i in base if f(key[i])]
    print(n,len(ids),"A==B",sum(a[i]["label"]==b[i]["label"] for i in ids),"A unk",sum(a[i]["label"]==U for i in ids),"B unk",sum(b[i]["label"]==U for i in ids),
     "both dec",sum(a[i]["label"]!=U!=b[i]["label"] for i in ids),"both dec agree",sum(a[i]["label"]==b[i]["label"]!=U for i in ids))
print("B label counts",C.Counter(r["label"] for r in b.values()).most_common())
print("A label counts",C.Counter(r["label"] for r in a.values()).most_common())
s=[r.get("seconds") for r in b.values() if r.get("seconds") is not None]
print("B sec <1s",sum(x<1 for x in s),"unsure",sum(bool(r.get("unsure")) for r in b.values()),"notes",[r.get("note") for r in b.values() if r.get("note")])
# drift: unknown rate and time by position quartile
for nm,x in (("A",a),("B",b)):
    rows=sorted(x.values(),key=lambda r:r.get("order",r.get("idx",0)) if isinstance(r.get("order",r.get("idx",0)),(int,float)) else 0)
    print(nm,"keys",list(rows[0].keys()))
