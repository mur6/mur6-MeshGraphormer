import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def pca_test():
    np.random.seed(0)

    n_data = 100
    cs = 1 / np.sqrt(2)
    x = np.linspace(-1, 1, n_data)
    y = np.random.randn(n_data) * np.cos(x * np.pi / 2) / 6
    x, y = cs * x - cs * y, cs * x + cs * y
    xp, yp = 0.25, 0.5

    X = np.array([x, y]).reshape(-1, 2)
    Xp = np.array([xp, yp]).reshape(-1, 2)

    # fig1, axes1 = plt.subplots(2, 2)
    # axes1[0, 0].scatter(X[:, 0], X[:, 1], marker='o', s=4)
    # axes1[0, 0].scatter(Xp[0, 0], Xp[0, 1], marker='x', s=40)

    # print("Xp original:{}".format(Xp))
    # # Xp original:[[0.25 0.5 ]]


    # for axis in axes1.ravel():
    #     axis.set_aspect('equal')
    #     axis.set_xlim(-1, 1)
    #     axis.set_ylim(-1, 1)

    # plt.show()


    fig2, axes2 = plt.subplots(1, 2)

    axes2[0].scatter(X[:, 0], X[:, 1], marker='o', s=4)
    axes2[0].scatter(Xp[0, 0], Xp[0, 1], marker='x', s=40)

    print("Xp original:{}".format(Xp))
    # Xp original:[[0.25 0.5 ]]

    pca = PCA(n_components=1).fit(X)
    X_trans = pca.transform(X)
    Xp_trans = pca.transform(Xp)

    print("Xp transfomed:{}".format(Xp_trans))
    # Xp transfomed:[[0.53055026]]

    X_inv = pca.inverse_transform(X_trans)
    Xp_inv = pca.inverse_transform(Xp_trans)
    axes2[1].scatter(X_inv[:, 0], X_inv[:, 1], marker='o', s=4)
    axes2[1].scatter(Xp_inv[0, 0], Xp_inv[0, 1], marker='x', s=40)

    print("Xp inversed:{}".format(Xp_inv))
    # Xp inversed:[[0.37112019 0.37919056]]

    for axis in axes2.ravel():
        axis.set_aspect('equal')
        axis.set_xlim(-1, 1)
        axis.set_ylim(-1, 1)

    print(pca.mean_)
    print(pca.components_)
    # plt.show()


import torch

def pca_test_1():
    # torch.Size([32, 3, 20]) torch.Size([32, 3]) torch.Size([32, 2, 3])

    vert_3d = torch.rand(32, 3, 20)
    pca_mean = torch.rand(32, 3)
    pca_components = torch.rand(32, 2, 3)
    normal_v = torch.cross(pca_components[:, 0], pca_components[:, 1], dim=1).unsqueeze(1)
    print(f"normal_v: {normal_v.shape}")
    # print(vert_3d)
    vert_3d = (torch.transpose(vert_3d, 2, 1))
    # print(x[0])
    #x - pca_mean
    pca_mean = pca_mean.unsqueeze(1)
    print(pca_mean.shape)
    print(vert_3d.shape)
    x = (vert_3d - pca_mean) * normal_v
    print(x[0])
    print(torch.sum(x[0], dim=1).shape)



# # a = torch.tensor([[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]])
# # b = torch.tensor([[-2.1763, -0.4713], [-0.6986,  1.3702]])

# a = torch.tensor([
#     [0.16311385, 0.65593925, 0.73697868],
#     [-0.46181902, 0.71085705, -0.53047663],
# ])
# a = torch.ones(2, 3)
# b = torch.zeros(2, 3)

# r = torch.cdist(a, b, p=2)
# print(r)

lis = [[-0.0021862344499278402, 0.010565901436267244, 0.0018521089061974277], [-0.00394100233885071, 0.010362483595044636, -0.0019235230667725145], [0.0004741666009443479, -0.007993476206163632, -0.003856810802395798], [-0.001226955629933038, -0.007007770920664359, -0.00680022189809365], [0.004828414059742353, -0.006123320266934478, 0.006339349380278109], [0.003698485343893068, -0.007678581763250402, 0.0030450904091069986], [0.003270372057272125, -0.008838040828175209, 0.0014514290264542734], [-0.002323160177543998, -0.004396692452800716, -0.007499645405597943], [-0.002380298844232076, -0.0055062622878014455, -0.008290930379566644], [0.0014012724590539288, -0.008635327554714188, -0.0023160727549523252], [0.004273381321626912, -0.0032250284949122705, 0.006940359475052532], [0.004760182673141947, -0.0008949541044423661, 0.009365521968280846], [0.002528877761693263, 0.004788482052546153, 0.008165234819797779], [-0.004434891266397345, 0.007686161909509648, -0.004573250188456474], [-0.005117429241613097, 0.005651514784911119, -0.0072268007151539485], [0.0038272265878378786, 0.0028219204476654415, 0.00967595329391481], [-0.0005559743633334752, 0.009657962834033479, 0.0046951513741543965], [-0.00408834381302263, 0.001509027745368761, -0.007595053545880609], [-0.004092320516757356, -0.0008440459618387006, -0.009029202982437652], [0.0012842317764057698, 0.008100046036351399, 0.007581313086070449]]
x = torch.tensor(lis)
print(x.pow(2).sum(dim=1).sqrt().mean())

c = torch.rand(32, 3)
d = torch.rand(32, 3)
r = torch.cat((c, d), dim=1)
print(r.shape, r)
