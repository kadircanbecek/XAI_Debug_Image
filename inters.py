import random

cats_lp = [4, 8, 15, 17, 13, 22, 2, 14, 9]
cats_mp = [20, 29, 1, 16, 24, 5, 12, 21, 11]
cats_hp = [6, 7, 26, 27, 28, 31, 3, 25, 0]
cows_lp = [15, 20, 22, 24, 29, 0, 8, 17, 21]
cows_mp = [13, 14, 25, 3, 6, 7, 2, 5, 1]
cows_hp = [12, 28, 31, 11, 9, 27, 26, 4, 16]
spiders_lp = [16, 26, 3, 27, 0, 25, 28, 31, 11]
spiders_mp = [4, 7, 6, 12, 21, 5, 9, 1, 2]
spiders_hp = [14, 13, 24, 20, 29, 8, 17, 22, 15]

cows_mp_hp = sorted(cows_mp + cows_hp)
spider_mp_hp = sorted(spiders_lp + spiders_mp)

comb = [s for s in cows_mp_hp if s in spider_mp_hp]

print(cows_mp_hp)
print(spider_mp_hp)
print(comb)

comb = [s for s in cows_mp if s in spiders_mp]
print(comb)
cnt = 0
for l in [cats_lp, cats_mp, cats_hp, cows_lp, cows_mp, cows_hp, spiders_lp, spiders_mp, spiders_hp]:
    print(",".join([str(c) for c in sorted(l)]), end="&")
    cnt+=1
    if cnt%3==0:
        print("\n")

range_ = [round((random.random() - 0.5) * 15, 2) for _ in range(25)]
print(range_)
range_p_i = [(i,v) for i,v in enumerate(range_) if v > 0]
print([i for i,_ in range_p_i])
print([i for _,i in range_p_i])

l = [0.32, 2.77, 4.23, 1.96, 4.72, 4.2, 6.82, 0.01, 0.5, 0.37, 7.0, 3.05, 5.15]
ss = sum(l)
print([round(li/ss,3) for li in l ])

comb = [s for s in cats_mp if s in spiders_mp]
print(comb)
comb = [s for s in cats_mp if s in spiders_hp]
print(comb)

comb = [s for s in cats_mp if s in cows_mp]
print(comb)
comb = [s for s in cats_mp if s in cows_hp]
print(comb)

comb = [s for s in cats_lp if s in cows_lp]
print(comb)
comb = [s for s in cows_hp if s in cats_mp or s in cats_hp]
print(comb)
comb = [s for s in cows_hp if s in spiders_mp or s in spiders_lp]
print(comb)