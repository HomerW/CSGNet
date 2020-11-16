import matplotlib.pyplot as plt
import numpy as np

# with open("ws.txt") as file:
#     ws_cds = file.readlines()
# ws_cds = [float(x.strip()) for x in ws_cds]
#
# with open("ws_st.txt") as file:
#     ws_st_cds = file.readlines()
# ws_st_cds = [float(x.strip()) for x in ws_st_cds]
#
# with open("ws_ab.txt") as file:
#     ws_ab_cds = file.readlines()
# ws_ab_cds = [float(x.strip()) for x in ws_ab_cds]
#
# with open("ws_st_ab.txt") as file:
#     ws_st_ab_cds = file.readlines()
# ws_st_ab_cds = [float(x.strip()) for x in ws_st_ab_cds]

# with open("ws_test.txt") as file:
#     wst_cds = file.readlines()
# wst_cds = [float(x.strip()) for x in wst_cds]

with open("ws_final_test.txt") as file:
    wst_cds = file.readlines()
wst_cds = [float(x.strip()) for x in wst_cds]

with open("ws_st_test.txt") as file:
    wst_st_cds = file.readlines()
wst_st_cds = [float(x.strip()) for x in wst_st_cds]

with open("ws_frozen_test.txt") as file:
    wstf_cds = file.readlines()
wstf_cds = [float(x.strip()) for x in wstf_cds]

with open("ws_frozen_st_test.txt") as file:
    wstf_st_cds = file.readlines()
wstf_st_cds = [float(x.strip()) for x in wstf_st_cds]

with open("ws_final_epochs.txt") as file:
    ws_epochs = file.readlines()
ws_epochs = [float(x.strip()) for x in ws_epochs]

with open("ws_st_epochs.txt") as file:
    ws_st_epochs = file.readlines()
ws_st_epochs = [float(x.strip()) for x in ws_st_epochs]

with open("ws_frozen_epochs.txt") as file:
    wsf_epochs = file.readlines()
wsf_epochs = [float(x.strip()) for x in wsf_epochs]

with open("ws_frozen_st_epochs.txt") as file:
    wsf_st_epochs = file.readlines()
wsf_st_epochs = [float(x.strip()) for x in wsf_st_epochs]

fig, ax = plt.subplots()

ws_epochs = [0] + ws_epochs
ws_st_epochs = [0] + ws_st_epochs
wsf_epochs = [0] + wsf_epochs
wsf_st_epochs = [0] + wsf_st_epochs

ax.plot(ws_epochs[:len(wst_cds)], wst_cds, 'b', label="LEST")
ax.plot(ws_st_epochs[:len(wst_st_cds)], wst_st_cds, 'g', label="self-training")
ax.plot(wsf_epochs[:len(wstf_cds)], wstf_cds, 'b--', label="LEST frozen")
ax.plot(wsf_st_epochs[:len(wstf_st_cds)], wstf_st_cds, 'g--', label="self-training frozen")

ax.legend()
ax.set_ylabel("chamfer distance")
ax.set_xlabel("epoch")
plt.show()
