import os
import numpy as np
from opfunu.cec_based import cec2020
from copy import deepcopy


PopSize = 100
DimSize = 10
LB = [-100] * DimSize
UB = [100] * DimSize
TrialRuns = 30

MaxFEs = DimSize * 1000
MaxIter = int(MaxFEs / PopSize)
curIter = 0
Pop = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)
Func_num = 0
SuiteName = "CEC2013"

Na = int(PopSize / 2)

# initialize the Pop randomly
def Initialization(func):
    global Pop, FitPop, DimSize
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = func.evaluate(Pop[i])


def ISAO(func):
    global Pop, FitPop, curIter, MaxIter, LB, UB, PopSize, DimSize, Na
    idx_sort = np.argsort(FitPop)
    Elites = [Pop[idx_sort[0]], Pop[idx_sort[1]], Pop[idx_sort[2]],
              np.mean(Pop[idx_sort[0:int(len(idx_sort) / 2)]], axis=0)]

    FitBest = FitPop[idx_sort[0]]
    X_mean = np.mean(Pop, axis=0)
    M = 0.35 + 0.25 * ((np.exp(curIter/MaxIter) - 1) / (np.e - 1)) * np.exp(-curIter/MaxIter)  # Snow melt rate
    RB = np.random.rand(DimSize)
    idx = list(range(PopSize))
    Na_idx = np.random.choice(idx, Na, replace=False)
    Nb_idx = [i for i in idx if i not in Na_idx]

    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)
    for i in Na_idx:  # Exploration
        k = np.random.randint(0, len(Elites))
        theta_1 = np.random.rand()
        for j in range(DimSize):
            Off[i][j] = Elites[k][j] + RB[j] * (theta_1 * (Elites[0][j] - Pop[i][j]) + (1 - theta_1) * (X_mean[j] - Pop[i][j]))
        Off[i] = np.clip(Off[i], LB, UB)
        FitOff[i] = func.evaluate(Off[i])
        if FitOff[i] < FitBest:
            FitBest = FitOff[i]
            Elites[0] = deepcopy(Off[i])

    for i in Nb_idx:  # Exploitation
        theta_2 = np.random.uniform(-1, 1)
        for j in range(DimSize):
            Off[i][j] = M * Elites[0][j] + RB[j] * (theta_2 * (Elites[0][j] - Pop[i][j]) + (1 - theta_2) * (X_mean[j] - Pop[i][j]))
        Off[i] = np.clip(Off[i], LB, UB)
        FitOff[i] = func.evaluate(Off[i])
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

def RunISAO(func):
    global curIter, MaxIter, TrialRuns, Pop, FitPop, DimSize, Na
    All_Trial_Best = []
    for i in range(TrialRuns):
        Best_list = []
        curIter = 0
        Na = int(PopSize / 2)
        Initialization(func)
        Best_list.append(min(FitPop))
        np.random.seed(2022 + 88 * i)
        while curIter < MaxIter:
            ISAO(func)
            Na = min(Na + 1, PopSize)
            curIter += 1
            Best_list.append(min(FitPop))
        All_Trial_Best.append(Best_list)
    np.savetxt("./ISAO_Data/" + SuiteName + "/F" + str(Func_num) + "_" + str(DimSize) + "D.csv", All_Trial_Best,
               delimiter=",")


def main(dim):
    global Func_num, DimSize, Pop, MaxFEs, SuiteName, LB, UB, MaxIter
    DimSize = dim
    Pop = np.zeros((PopSize, dim))
    MaxFEs = dim * 1000
    MaxIter = int(MaxFEs / DimSize)
    LB = [-100] * dim
    UB = [100] * dim

    CEC2020Funcs = [cec2020.F12020(dim), cec2020.F22020(dim), cec2020.F32020(dim), cec2020.F42020(dim),
                    cec2020.F52020(dim), cec2020.F62020(dim), cec2020.F72020(dim), cec2020.F82020(dim),
                    cec2020.F92020(dim), cec2020.F102020(dim)]

    SuiteName = "CEC2020"
    for i in range(len(CEC2020Funcs)):
        Func_num = i + 1
        RunISAO(CEC2020Funcs[i])


if __name__ == "__main__":
    if os.path.exists('ISAO_Data/CEC2020') == False:
        os.makedirs('ISAO_Data/CEC2020')
    Dims = [10, 30, 50, 100]
    for Dim in Dims:
        main(Dim)
