"""How noisy is the 8-sample pass rate? Compare it to the 16- and 32-sample estimates (same starting policy, temperature 1.0).
   python3 scripts/passrate_compare.py
"""
import json

D = "/home/ravi.sah/oml/outputs/difficulty"
load = lambda f: {r["uid"]: r["pass_rate"] for r in map(json.loads, open(f"{D}/{f}"))}
p8, p16, p32 = load("math.jsonl"), load("math16.jsonl"), load("math32.jsonl")
u = sorted(set(p8) & set(p16) & set(p32))
print(f"{len(u)} problems scored with 8, 16 and 32 samples\n")
for name, p in (("8", p8), ("16", p16), ("32", p32)):
    v = [p[x] for x in u]
    print(f"n={name:>2}: never solved (p=0): {sum(x == 0 for x in v):5d}   always solved (p=1): {sum(x == 1 for x in v):5d}   "
          f"in between (filter keeps): {sum(0 < x < 1 for x in v):5d}   mid band 3/8-5/8: {sum(0.375 - 1e-9 <= x <= 0.625 + 1e-9 for x in v):5d}")
z = [x for x in u if p8[x] == 0]
o = [x for x in u if p8[x] == 1]
print(f"\nProblems scored 0/8: {len(z)}. With 32 samples: {sum(p32[x] > 0 for x in z)} ({sum(p32[x] > 0 for x in z) / len(z):.0%}) turn out solvable (mean p32 among them {sum(p32[x] for x in z if p32[x] > 0) / max(1, sum(p32[x] > 0 for x in z)):.3f})")
print(f"Problems scored 8/8: {len(o)}. With 32 samples: {sum(p32[x] < 1 for x in o)} ({sum(p32[x] < 1 for x in o) / len(o):.0%}) are not always solved (mean p32 {sum(p32[x] for x in o) / len(o):.3f})")
m8 = [x for x in u if 0.375 - 1e-9 <= p8[x] <= 0.625 + 1e-9]
print(f"Mid band by 8 samples: {len(m8)}; of those only {sum(0.375 - 1e-9 <= p32[x] <= 0.625 + 1e-9 for x in m8)} are in the mid band by 32 samples")
mean_abs = sum(abs(p8[x] - p32[x]) for x in u) / len(u)
print(f"Mean |p8 - p32| = {mean_abs:.3f}")
