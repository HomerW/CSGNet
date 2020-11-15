import matplotlib.pyplot as plt
import numpy as np

with open("ws.txt") as file:
    ws_cds = file.readlines()
ws_cds = [float(x.strip()) for x in ws_cds]

with open("ws_st.txt") as file:
    ws_st_cds = file.readlines()
ws_st_cds = [float(x.strip()) for x in ws_st_cds]

with open("ws_ab.txt") as file:
    ws_ab_cds = file.readlines()
ws_ab_cds = [float(x.strip()) for x in ws_ab_cds]

with open("ws_st_ab.txt") as file:
    ws_st_ab_cds = file.readlines()
ws_st_ab_cds = [float(x.strip()) for x in ws_st_ab_cds]

fig, ax = plt.subplots()

ax.plot(ws_cds, label="LEST")
ax.plot(ws_st_cds, label="self-training")
ax.plot(ws_ab_cds, label="LEST top 5")
ax.plot(ws_st_ab_cds, label="self-training top 5")
ax.legend()
ax.set_ylabel("chamfer distance")
ax.set_xlabel("round")
plt.show()
