import matplotlib.pyplot as plt

# fig, ax = plt.subplots()
gs_kw = dict(height_ratios=[1, 5])
f, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw=gs_kw, sharex=True)

with open("ws.txt") as file:
    ws_cds = file.readlines()
ws_cds = [float(x.strip()) for x in ws_cds]

with open("ws-epochs.txt") as file:
    ws_inf = file.readlines()
ws_inf = [float(x.strip()) for x in ws_inf]

with open("ws-epochs-gen.txt") as file:
    ws_gen = file.readlines()
ws_gen = [float(x.strip()) for x in ws_gen]

with open("ws-simple.txt") as file:
    simple_cds = file.readlines()
simple_cds = [float(x.strip()) for x in simple_cds]

with open("ws-simple-epochs.txt") as file:
    simple_inf = file.readlines()
simple_inf = [float(x.strip()) for x in simple_inf]

with open("ws-switch.txt") as file:
    switch_cds = file.readlines()
switch_cds = [float(x.strip()) for x in switch_cds]

with open("ws-switch-epochs.txt") as file:
    switch_inf = file.readlines()
switch_inf = [float(x.strip()) for x in switch_inf]

with open("ws-switch-epochs-gen.txt") as file:
    switch_gen = file.readlines()
switch_gen = [float(x.strip()) for x in switch_gen]

with open("ws-simple-reset.txt") as file:
    reset_cds = file.readlines()
reset_cds = [float(x.strip()) for x in reset_cds]

with open("ws-simple-reset-epochs.txt") as file:
    reset_inf = file.readlines()
reset_inf = [float(x.strip()) for x in reset_inf]

switch_inf = [x / 100 for x in switch_inf]

ws_time = [(inf*43 + gen*2.7 + 402) / 60 / 60 for inf, gen in zip(ws_inf, ws_gen)]
simple_time = [(inf*43 + 402) / 60 / 60 for inf in simple_inf]
switch_time = [(inf*43 + gen*2.7 + 402) / 60 / 60 for inf, gen in zip(switch_inf, switch_gen)]
reset_time = [(inf*43 + 402) / 60 / 60 for inf in reset_inf]

ws_time = [0] + ws_time
simple_time = [0] + simple_time
switch_time = [0] + switch_time
reset_time = [0] + reset_time

print(reset_time[-1])

ax2.set_ylabel("Chamfer Distance on Cad Validation")
ax2.set_xlabel("Time (hours)")
ax.set_title("Performance of Wake Sleep Variants")

ax.plot(ws_time, ws_cds, 'r', label="Wake Sleep, Converge Switch")
ax.plot(simple_time, simple_cds, 'b', label="Wake Sleep, No Gen. Model")
ax.plot(switch_time[:-1], switch_cds, 'c', label="Wake Sleep, 1 Epoch Switch")
ax.plot(reset_time, reset_cds, 'y', label="Wake Sleep, No Gen. Model, Reset Weights")
ax.plot(range(15), [1.14] * 15, 'g--', label="SL + RL (best CD)")
ax.plot(range(15), [ws_cds[0]] * 15, 'g', label="SL (best CD)")

ax2.plot(ws_time, ws_cds, 'r', label="Wake Sleep, Converge Switch")
ax2.plot(simple_time, simple_cds, 'b', label="Wake Sleep, No Gen. Model")
ax2.plot(switch_time[:-1], switch_cds, 'c', label="Wake Sleep, 1 Epoch Switch")
ax2.plot(reset_time, reset_cds, 'y', label="Wake Sleep, No Gen. Model, Reset Weights")
ax2.plot(range(15), [1.14] * 15, 'g--', label="SL + RL (best CD)")
ax2.plot(range(15), [ws_cds[0]] * 15, 'g', label="SL (best CD)")
leg = ax2.legend()

# zoom-in / limit the view to different portions of the data
ax.set_ylim(1.7, 4)  # outliers only
ax2.set_ylim(1, 1.6)  # most of the data

# hide the spines between ax and ax2
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-5*d, +5*d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-5*d, +5*d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

plt.draw() # Draw the figure so you can find the positon of the legend.

# Get the bounding box of the original legend
bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)

# Change to location of the legend.
yOffset = .7
bb.y0 += yOffset
bb.y1 += yOffset
leg.set_bbox_to_anchor(bb, transform = ax.transAxes)

plt.savefig("cd_graph_time.png")

# regular_fid = [
#  0.980387559465866,
#  0.6138332072750277,
#  0.6622565361034043,
#  0.6588748396095208,
#  0.6670089702070827,
#  0.6622776339560645,
#  0.6025977758832246,
#  0.5466552587508293,
#  0.5799835974838905,
#  0.5845467203891186,
#  0.5867330861382867,
#  0.5776176210836739,
#  0.5376165712417362,
#  0.5224034586129609,
#  0.569011531418754,
#  0.545484496955668,
#  0.5763107812139345,
#  0.614424305987415,
#  0.578352824440441,
#  0.5242809088406577,
#  0.5548492738143724,
#  0.5257463931140565,
#  0.5326157406103353,
#  0.5631644328337801,
#  0.5483919546612701,
#  0.5125295521001678,
#  0.5440680884742792,
#  0.5074970372228218,
#  0.540705218045866
# ]
#
# tree_fid = [
#   2.303648249132067,
#   1.2861168947518586,
#   0.8548721602888265,
#   1.0035051537384398,
#   1.0484782601146119,
#   0.9000103150322594,
#   0.7402756472881444,
#   0.6767120642102562,
#   0.5983757789949309,
#   0.5907863242658584,
#   0.7106369493409908,
#   0.7485950009663815,
#   0.5485656157357277,
#   0.5302019192184435,
#   0.6032816695455421,
#   0.444865301277074,
#   0.4286765691725052,
#   0.6674036620605235,
#   0.4475728295945032,
#   0.745844090017171,
#   0.391216621921304,
#   0.5115365489401742,
#   0.42490519403277305,
#   0.5071544730962405,
#   0.433595708640484,
#   0.48481947918337176
# ]
#
# regular_x = [x/len(regular_fid) for x in range(len(regular_fid))]
# tree_x = [x/len(tree_fid) for x in range(len(tree_fid))]
#
# ax.plot(regular_x, regular_fid, 'r', label="sequence vae")
# ax.plot(tree_x, tree_fid, 'b', label="tree vae")
# ax.set_ylabel("FID")
# ax.set_xlabel("time")
# ax.set_title("sequence vs tree FID, activations and inferred programs from best model, 1 hour")
# ax.legend()
# plt.savefig("fid_graph.png")
