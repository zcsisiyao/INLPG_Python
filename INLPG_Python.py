import numpy as np
from scipy.spatial.ckdtree import cKDTree
import matplotlib.pyplot as plt
import pywt
import cv2


def imageTodata(x, ps, ss):
    """
    Image is converted into patch.
    """

    H, W, B = x.shape
    i_x = list(range(ps, H-ps-1, ss))
    if i_x[-1] != H-ps-1:
        i_x.append(H-ps-1)
    j_y = list(range(ps, W-ps-1, ss))
    if j_y[-1] != W-ps-1:
        j_y.append(W-ps-1)
    PGM = []
    for i in range(len(i_x)):
        for j in range(len(j_y)):
            i_x_d = i_x[i] - ps
            i_x_u = i_x[i] + ps
            j_y_d = j_y[j] - ps
            j_y_u = j_y[j] + ps
            data = x[i_x_d:i_x_u+1, j_y_d:j_y_u+1, :]
            data = np.reshape(data, -1)
            PGM.append(data)
    return PGM

def local_index(i, ps, ss, H):
    """
    Position switch.
    """

    if i < ss:
        local_x = ps
    elif i >= np.floor((H-2*ps-1)/ss)*ss+2*ps:
        local_x = H-ps-1
    else:
        a = ps+max(0, np.ceil((i-2*ps-1)/ss)) *ss
        b = min((ps+np.floor((i)/ss)*ss), H)
        local_x = list(range(int(a), int(b), ss))
    return local_x


def dataToimage(data, ps, ss, image):
    """
    Patch is converted into image.
    """

    H, W, _ = image.shape
    image_re = np.zeros((H, W))
    i_x = list(range(ps, H-ps-1, ss))
    if i_x[-1] != H-ps-1:
        i_x.append(H-ps-1)
    j_y = list(range(ps, W-ps-1, ss))
    if j_y[-1] != W-ps-1:
        j_y.append(W-ps-1)
    k = 0
    for i in range(len(i_x)):
        for j in range(len(j_y)):
            image_re[i_x[i], j_y[j]] = data[k]
            k += 1

    DI = np.zeros_like(image_re)
    for i in range(H):
        for j in range(W):
            local_i = local_index(i, ps, ss, H)
            local_j = local_index(j, ps, ss, W)
            DI[i, j] = np.mean(np.mean(image_re[local_i, local_j]))
    return DI


def KNNsearch(x, y, k):
    """
    Search k neighbor for similar patches.
    """

    YourTreeName = cKDTree(x)
    id_x, dist_x = [], []
    for item in y:
        dist, id = YourTreeName.query(item, k)
        id_x.append(id)
        dist_x.append(dist)
    return np.array(id_x), np.array(dist_x)

def INLPG_mappedKNN(x, y, k):
    """
    Computing patch differences.
    """
    k += 1
    print("------knnsearch_start!!!------")
    idx, distX = KNNsearch(x, x, k)
    print("------knnsearch_half_finshed!!!------")
    idy, distY = KNNsearch(y, y, k)
    print("------knnsearch_finshed!!!------")
    N, D = x.shape
    fx_dist = np.zeros((N, 1))
    fy_dist = np.zeros((N, 1))
    fx_dim = np.zeros((N, 1))
    fy_dim = np.zeros((N, 1))
    for i in range(N):
        di_x = distX[i, 1:k]
        id_x = idx[i, 1:k]
        di_y = distY[i, 1:k]
        id_y = idy[i, 1:k]
        di_x_y, di_y_x = [], []
        for m in range(1, k):
            di_x_y.append(np.linalg.norm(x[i, :] - x[idy[i, m], :], ord=2))
            di_y_x.append(np.linalg.norm(y[idx[i, m], :] - y[i, :], ord=2))
        fx_dist[i] = abs(np.mean(di_x) - np.mean(di_x_y))
        fy_dist[i] = abs(np.mean(di_y) - np.mean(di_y_x))
    return fx_dist, fy_dist

def creat_gauss_kernel(kernel_size=3, sigma=1, k=1):
    """
    Creating gauss kernel.
    reference: https://blog.csdn.net/jasneik/article/details/108150217
    """

    if sigma == 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
    X = np.linspace(-k, k, kernel_size)
    Y = np.linspace(-k, k, kernel_size)
    x, y = np.meshgrid(X, Y)
    x0 = 0
    y0 = 0
    gauss = 1/(2*np.pi*sigma**2) * np.exp(- ((x -x0)**2 + (y - y0)**2)/ (2 * sigma**2))
    return gauss

def DWT_fusionforDI(DIbw, DIfw, p):
    """
    Difference image fusion: two-dimensional discrete wavelet fusion generates difference image.
    """

    cA1, (cH1, cV1, cD1) = pywt.dwt2(DIbw,wavelet='db4')
    cA2, (cH2, cV2, cD2) = pywt.dwt2(DIfw, wavelet='db4')
    alfa = 0.5
    cAf = alfa * cA1 + (1 - alfa) * cA2
    p  = 2 * p + 1
    h = creat_gauss_kernel(kernel_size=p,sigma=1, k=1)
    eH1 = cv2.filter2D(cH1 * cH1, -1, h)
    eV1 = cv2.filter2D(cV1 * cV1, -1, h)
    eD1 = cv2.filter2D(cD1 * cD1, -1, h)
    eH2 = cv2.filter2D(cH1 * cH2, -1, h)
    eV2 = cv2.filter2D(cV1 * cV2, -1, h)
    eD2 = cv2.filter2D(cD1 * cD2, -1, h)
    cHf = (np.sign(eH1 - eH2) + 1) * cH2 / 2 + (np.sign(eH2-eH1) + 1) * cH1 / 2
    cVf = (np.sign(eV1 - eV2) + 1) * cV2 / 2 + (np.sign(eV2 - eV1) + 1) * cV1 / 2
    cDf = (np.sign(eD1 - eD2) + 1) * cD2 / 2 + (np.sign(eD2 - eD1) + 1) * cD1 / 2
    DI_fuse = pywt.idwt2((cAf, (cHf, cVf, cDf)), wavelet='db4')
    return DI_fuse

def main():
    """
    Load dataset.
    data type: tensor
    x: pre change image
    y: post change image
    gt: ground truth
    ps: patch size: ps * 2 + 1
    ss: stride size
    """

    x, y, gt = _bern()
    x, y = x.numpy(), y.numpy()
    print("------Loading dataset--Finshed!!!------")
    # patch group matrix
    PGM_x = imageTodata(x, ps=2, ss=2)
    PGM_x = np.array(PGM_x)
    PGM_y = imageTodata(y, ps=2, ss=2)
    PGM_y = np.array(PGM_y)
    print("------PGM--Finshed!!!------")
    # structure difference calulation
    k = round(x.shape[0] * 0.1)
    print(f"k:{k}")
    [fx_dist, fy_dist] = INLPG_mappedKNN(PGM_x, PGM_y, k)
    # DI generation
    print("------MAPPING--Finshed!!!------")
    DIbw_dist = dataToimage(fx_dist, ps=2, ss=2, image=x)
    DIfw_dist = dataToimage(fy_dist, ps=2, ss=2, image=x)
    DIfusion_dist = DWT_fusionforDI(DIbw_dist, DIfw_dist, 2)
    plt.imshow(DIfusion_dist, cmap='gray')
    plt.show()
if __name__=="__main__":
    main()

