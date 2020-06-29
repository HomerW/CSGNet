import matplotlib.pyplot as plt

fig, ax = plt.subplots()

cds = [1.730185917301833,
       1.549893241731138,
       1.457912776081185,
       1.3866184009993416,
       1.320071422674157,
       1.2986928377967961,
       1.2830639292180508,
       1.2660030433692226,
       1.2709239809915438,
       1.2620423701167842,
       1.269719117838698,
       1.2465354029063833,
       1.2386172573544962,
       1.2392223470031016,
       1.2262864245587017,
       1.2360737861320252,
       1.2492554585176665,
       1.2244267643360767,
       1.2163007819205056,
       1.2371063688654722,
       1.2202888888787544]

ax.set_ylabel("chamfer distance")
ax.set_xlabel("iteration")
ax.set_xticks(range(len(cds)))
ax.set_title("wake sleep inference c.d, no generator, b.w. = 5, patience = 5")

ax.plot(range(len(cds)), cds)
plt.savefig("cd_graph1.png")
