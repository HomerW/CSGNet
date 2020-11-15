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

with open("ws_test.txt") as file:
    wst_cds = file.readlines()
wst_cds = [float(x.strip()) for x in wst_cds]

with open("ws_st_test.txt") as file:
    wst_st_cds = file.readlines()
wst_st_cds = [float(x.strip()) for x in wst_st_cds]

with open("ws_ab_test.txt") as file:
    wst_ab_cds = file.readlines()
wst_ab_cds = [float(x.strip()) for x in wst_ab_cds]

with open("ws_st_ab_test.txt") as file:
    wst_st_ab_cds = file.readlines()
wst_st_ab_cds = [float(x.strip()) for x in wst_st_ab_cds]

fig, ax = plt.subplots()

ax.plot(ws_cds, 'b', label="LEST")
ax.plot(ws_st_cds, 'g', label="self-training")
ax.plot(ws_ab_cds, 'r', label="LEST top 5")
ax.plot(ws_st_ab_cds, 'c', label="self-training top 5")

ax.plot(range(1, len(wst_cds) + 1), wst_cds, "b--", label="LEST test")
ax.plot(range(1, len(wst_st_cds) + 1), wst_st_cds, "g--", label="self-training test")
ax.plot(range(1, len(wst_ab_cds) + 1), wst_ab_cds, "r--", label="LEST top 5 test")
ax.plot(range(1, len(wst_st_ab_cds) + 1), wst_st_ab_cds, "c--", label="self-training top 5 test")

ax.legend()
ax.set_ylabel("chamfer distance")
ax.set_xlabel("round")
plt.show()
