import matplotlib.pyplot as plt

with open("cad-cd.txt") as file:
    cds = file.readlines()
cds = [float(x.strip()) for x in cds]

fig, ax = plt.subplots()

ax.plot([(x * 362) / 60 / 60 for x in range(len(cds))], cds)

ax.set_ylabel("Chamfer Distance on Cad Validation")
ax.set_xlabel("Time (hours)")
ax.set_title("RL")

plt.savefig('cad_graph.png')
