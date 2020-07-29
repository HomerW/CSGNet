import matplotlib.pyplot as plt

fig, ax = plt.subplots()

with open("cont-cd.txt") as file:
    cont_cds = file.readlines()
cont_cds = [float(x.strip()) for x in cont_cds]

with open("disc-cd.txt") as file:
    disc_cds = file.readlines()
disc_cds = [float(x.strip()) for x in disc_cds]

disc_cds = disc_cds[:76]

ax.set_ylabel("chamfer distance")
ax.set_xlabel("epoch")
ax.set_title("discrete and continuous inference net chamfer distance")

ax.plot(range(76), cont_cds, 'r', label="continuous")
ax.plot(range(76), disc_cds, 'b', label="discrete")
ax.legend()
plt.savefig("cd_graph2.png")

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
