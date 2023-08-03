import os
from copy import deepcopy
import numpy as np
import cv2 as cv


PopSize = 30
DimSize = 10
LB = [0] * DimSize
UB = [256] * DimSize
TrialRuns = 30
epsilon = 1e-10
MaxIter = 25 * DimSize

curIter = 0

Pop = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)

Func_num = str(0)
SuiteName = "Segmentation"

X_best = np.zeros(DimSize)
FitBest = float("inf")
Na = int(PopSize / 2)

HIST = None


def Otsu(indi):
    global HIST
    """Given a list of indi and an image location , returns the fitness using Otsu's Objective function"""
    tmpIndi = list(map(int, sorted(indi)))
    tmpIndi.append(256)
    tmpIndi.insert(0, 0)

    cumulative_sum = []
    cumulative_mean = []
    global_mean = 0
    Sigma = 0

    for i in range(len(tmpIndi)-1):
        cumulative_sum.append(sum(HIST[tmpIndi[i]:tmpIndi[i + 1]]) + epsilon)  # Cumulative sum of each Class
        cumulative = 0
        for j in range(tmpIndi[i], tmpIndi[i + 1]):
            cumulative = cumulative + (j + 1) * HIST[j]
        cumulative_mean.append(cumulative / cumulative_sum[-1])  # Cumulative mean of each Class
        global_mean = global_mean + cumulative  # Global Intensity Mean
    for i in range(len(cumulative_mean)):  # Computing Sigma
        Sigma = Sigma + (cumulative_sum[i] * ((cumulative_mean[i] - global_mean) ** 2))
    return -Sigma[0]


# initialize the Pop randomly
def Initialization():
    global Pop, FitPop, DimSize, X_best, FitBest
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = Otsu(Pop[i])
    FitBest = min(FitPop)
    X_best = Pop[np.argmin(FitPop)]


def ISAO():
    global Pop, FitPop, curIter, MaxIter, LB, UB, PopSize, DimSize, X_best, FitBest
    idx_sort = np.argsort(FitPop)

    Elites = [X_best, Pop[idx_sort[1]], Pop[idx_sort[2]], np.mean(Pop[idx_sort[0:int(len(idx_sort)/2)]], axis=0)]
    X_mean = np.mean(Pop, axis=0)

    M = 0.35 + 0.25 * ((np.exp(curIter/MaxIter) - 1) / (np.e - 1)) * np.exp(-curIter/MaxIter)  # Snow melt rate

    RB = np.random.rand(DimSize)  # Brownian random number vector
    idx = list(range(PopSize))
    Na_idx = np.random.choice(idx, Na, replace=False)
    Nb_idx = [i for i in idx if i not in Na_idx]
    FitBest = FitPop[idx_sort[0]]

    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)
    for i in Na_idx:  # Exploration
        k = np.random.randint(0, len(Elites))
        theta_1 = np.random.rand()
        for j in range(DimSize):
            Off[i][j] = Elites[k][j] + RB[j] * (
                        theta_1 * (Elites[0][j] - Pop[i][j]) + (1 - theta_1) * (X_mean[j] - Pop[i][j]))
        Off[i] = np.clip(Off[i], LB, UB)
        FitOff[i] = Otsu(Off[i])
        if FitOff[i] < FitBest:
            FitBest = FitOff[i]
            Elites[0] = deepcopy(Off[i])

    for i in Nb_idx:  # Exploitation
        theta_2 = np.random.uniform(-1, 1)
        for j in range(DimSize):
            Off[i][j] = M * Elites[0][j] + RB[j] * (
                        theta_2 * (Elites[0][j] - Pop[i][j]) + (1 - theta_2) * (X_mean[j] - Pop[i][j]))
        Off[i] = np.clip(Off[i], LB, UB)
        FitOff[i] = Otsu(Off[i])
        if FitOff[i] < FitBest:
            FitBest = FitOff[i]
            Elites[0] = deepcopy(Off[i])

    tempIndi = np.vstack((Pop, Off))
    tempFit = np.hstack((FitPop, FitOff))
    tmp = list(map(list, zip(range(len(tempFit)), tempFit)))
    small = sorted(tmp, key=lambda x: x[1], reverse=False)
    for i in range(PopSize):
        key, _ = small[i]
        FitPop[i] = tempFit[key]
        Pop[i] = tempIndi[key].copy()
    X_best = deepcopy(Pop[np.argmin(FitPop)])


def RunISAO():
    global curIter, MaxIter, TrialRuns, Pop, FitPop, DimSize, FitBest, Na
    All_Trial_Best = []
    Best_Variable = []
    for i in range(TrialRuns):
        Best_list = []
        curIter = 0
        Initialization()
        Na = int(PopSize / 2)
        Best_list.append(FitBest)
        np.random.seed(2022 + 88 * i)
        while curIter < MaxIter:
            ISAO()
            Na = min(Na + 1, PopSize)
            curIter += 1
            Best_list.append(FitBest)
        Best_Variable.append(list(map(int, sorted(X_best))))
        All_Trial_Best.append(np.abs(Best_list))
    np.savetxt("./ISAO_Data/Segmentation/Fitness/" + Func_num + "_" + str(DimSize) + "D.csv", All_Trial_Best, delimiter=",")
    np.savetxt("./ISAO_Data/Segmentation/Solution/" + Func_num + "_" + str(DimSize) + "D.csv", Best_Variable, delimiter=",")


def main(dim):
    global Func_num, DimSize, Pop, MaxIter, SuiteName, LB, UB, HIST

    SuiteName = "Segmentation"
    Pics = list(range(1, 9))
    for pic in Pics:
        DimSize = dim
        Pop = np.zeros((PopSize, DimSize))
        LB = np.array([0] * dim)
        UB = np.array([256] * dim)
        MaxIter = 10 * dim

        img_path = '../Image/pic' + str(pic) + '.jpg'
        img = cv.imread(img_path, 0)
        pixels = cv.calcHist([img], [0], None, [256], [0, 256])
        HIST = pixels / (img.shape[0] * img.shape[1])

        Func_num = str(pic)
        RunISAO()


if __name__ == "__main__":
    if os.path.exists('./ISAO_Data/Segmentation/Fitness') == False:
        os.makedirs('./ISAO_Data/Segmentation/Fitness')
    if os.path.exists('./ISAO_Data/Segmentation/Solution') == False:
        os.makedirs('./ISAO_Data/Segmentation/Solution')
    Dims = [4, 8, 12]
    for dim in Dims:
        main(dim)
