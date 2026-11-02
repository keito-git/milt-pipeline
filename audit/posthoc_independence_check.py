import json,sys,collections as C
import numpy as np
from scipy.stats import spearmanr
R=sys.argv[1]
key={r["id"]:r for r in json.load(open("audit_key.json"))["items"]}
A,B=[json.load(open(f"{R}/auditor{i}.json")) for i in (1,2)]
a,b=[{r["id"]:r for r in x["answers"]} for x in (A,B)]
ids=list(key)
print("position spearman",spearmanr([a[i]["shown_position"] for i in ids],[b[i]["shown_position"] for i in ids]))
print("seconds spearman",spearmanr([a[i]["seconds"] for i in ids],[b[i]["seconds"] for i in ids]))
print("changed A",sum(bool(a[i].get("changed")) for i in ids),"B",sum(bool(b[i].get("changed")) for i in ids))
U="不明"
for nm,x in (("A",a),("B",b)):
    s=sorted(x.values(),key=lambda r:r["shown_position"])
    sec=np.array([r["seconds"] for r in s]);lab=[r["label"] for r in s]
    print(nm,"sec pct",np.percentile(sec,[5,25,50,75,95]).round(2),"max",sec.max(),"sum h",round(sec.sum()/3600,2))
    print(" by quartile: med sec",[round(float(np.median(sec[j:j+224])),2) for j in range(0,896,224)],"unk",[lab[j:j+224].count(U) for j in range(0,896,224)])
    print(" P(same as prev)",round(np.mean([p==q for p,q in zip(lab,lab[1:])]),3),"expected",round(sum((v/len(lab))**2 for v in C.Counter(lab).values()),3))
base=[i for i in ids if not key[i]["is_hidden_repeat"]]
dec=[i for i in base if a[i]["label"]!=U and b[i]["label"]!=U]
hard=[i for i in dec if key[i]["band"]!="[0.9,1.0]"]
print("both decided",len(dec),"disagree:",[(key[i]["auto_label"],key[i]["gold"],a[i]["label"],b[i]["label"]) for i in dec if a[i]["label"]!=b[i]["label"]])
print("hard decided",len(hard),"agree",sum(a[i]["label"]==b[i]["label"] for i in hard))
nongold=[i for i in dec if a[i]["label"]!=key[i]["gold"]]
print("both decided, A!=gold",len(nongold),"of which B==A",sum(a[i]["label"]==b[i]["label"] for i in nongold))
print(" those",[(key[i]["band"],key[i]["auto_label"],key[i]["gold"],a[i]["label"],b[i]["label"]) for i in nongold])
nonauto=[i for i in dec if a[i]["label"]!=key[i]["auto_label"]]
print("both decided, A!=auto",len(nonauto),"B==A",sum(a[i]["label"]==b[i]["label"] for i in nonauto))
fast=[i for i in dec if b[i]["seconds"]<1]
print("B<1s decided",len(fast),"agree A",sum(a[i]["label"]==b[i]["label"] for i in fast),"agree gold",sum(b[i]["label"]==key[i]["gold"] for i in fast))
print("gold agreement base: A",sum(a[i]["label"]==key[i]["gold"] for i in base),"B",sum(b[i]["label"]==key[i]["gold"] for i in base),"of",len(base))
print("unknown in A: gold dist",C.Counter(key[i]["gold"] for i in base if a[i]["label"]==U).most_common(8))
print("unknown in A: B dist",C.Counter(b[i]["label"] for i in base if a[i]["label"]==U).most_common(8))
print("gold labels dist",C.Counter(key[i]["gold"] for i in base).most_common())
rep=[i for i in ids if key[i]["is_hidden_repeat"]]
print("repeat strata",C.Counter(key[i]["stratum"] for i in rep),"gold",C.Counter(key[i]["gold"] for i in rep).most_common(5))
