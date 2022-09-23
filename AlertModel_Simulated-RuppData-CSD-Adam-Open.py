import numpy as np
import pandas as pd
import math
import scipy
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from timeit import default_timer as timer
import copy

PI=3.1415926535
R_c = 2880
LowBound=0.87
a_s=0.235
A_S=2400*(1+a_s)/R_c


# SAFTE model: Performance：
def fx(X, theta, S_WK, S_SP):
    RT = theta[0]
    A_h_0 = theta[1]
    K = theta[2]
    tau_d = theta[3]
    d_1 = theta[4]
    d_2 = theta[5]
    p = theta[6]
    p_ = theta[7]
    beta = theta[8]

    y_pre = np.zeros(len(X))

    for i in range(len(X)):
        t = X[i]
        for u in range(1, 12):
            if (SP_time[u - 1] <= t and t <= WK_time[u]):
                I = u
                Type = 'W'
                Start_time_I = SP_time[u - 1]
                break
            if (WK_time[u] <= t and t < SP_time[u]):
                I = u
                Type = 'S'
                Start_time_I = WK_time[u]
                break

        if (Type == 'W'):
            Rt = S_SP[I - 1] - K * (t - Start_time_I)
        else:
            Rt = A_S - math.exp(-(t - Start_time_I) / tau_d) * (A_S - S_WK[I])

        fx = (Rt) * (RT * 1 - d_2 * (math.cos(2 * PI * (t - p) / 24) + beta * math.cos(4 * PI * (t - p - p_) / 24))) + (
                    d_1 + d_2) * (math.cos(2 * PI * (t - p) / 24) + beta * math.cos(4 * PI * (t - p - p_) / 24))

        y_pre[i] = fx
    return y_pre


# calculate L2 loss function：
def Recurr_computeCost(X, Y, theta, i, S_WK, S_SP):
    RT = theta[0]
    A_h_0 = theta[1]
    K = theta[2]
    tau_d = theta[3]
    d_1 = theta[4]
    d_2 = theta[5]
    p = theta[6]
    p_ = theta[7]
    beta = theta[8]

    t = X[i]
    for u in range(1, 12):
        if (SP_time[u - 1] <= t and t <= WK_time[u]):
            I = u
            Type = 'W'
            Start_time_I = SP_time[u - 1]
            break
        if (WK_time[u] <= t and t < SP_time[u]):
            I = u
            Type = 'S'
            Start_time_I = WK_time[u]
            break

    if (Type == 'W'):
        Rt = S_SP[I - 1] - K * (t - Start_time_I)
    else:
        Rt = A_S - math.exp(-(t - Start_time_I) / tau_d) * (A_S - S_WK[I])

    Ct = math.cos(2 * PI * (t - p) / 24) + beta * math.cos(4 * PI * (t - p - p_) / 24)

    Pt = (Rt * (RT * 1 - d_2 * Ct) + (d_1 + d_2) * Ct)

    inner = np.power((Pt - Y[i]), 2)
    return (float(inner / 2))


# calculate S_WK,S_SP,S_xi_WK,S_xi_SP,S_K_WK,S_K_SP,S_tau_SP,S_tau_WK
def calculated_periods_end_homestatic_level(theta, WK_len, SP_len):
    RT = theta[0]
    xi = theta[1]
    K = theta[2]
    tau_d = theta[3]
    d_1 = theta[4]
    d_2 = theta[5]
    p = theta[6]
    p_ = theta[7]
    beta = theta[8]

    S_SP = [xi]
    S_WK = [0]
    S_xi_WK = [0]
    S_xi_SP = [1]
    S_K_WK = [0]
    S_K_SP = [0]
    S_tau_SP = [0]
    S_tau_WK = [0]
    for i in range(1, 12):
        S_xi_WK.append(S_xi_SP[i - 1])

        S_xi_SP.append(math.exp(-(SP_len[i]) / tau_d) * S_xi_WK[i])

        S_K_WK.append(-WK_len[i] + S_K_SP[i - 1])

        S_K_SP.append(math.exp(-SP_len[i] / tau_d) * S_K_WK[i])

        S_WK.append(S_SP[i - 1] - K * WK_len[i])

        S_SP.append(A_S - math.exp(-SP_len[i] / tau_d) * (A_S - S_WK[i]))

        S_tau_WK.append(S_tau_SP[i - 1])

        S_tau_SP.append(math.exp(-SP_len[i] / tau_d) * (
                -SP_len[i] / tau_d / tau_d * (A_S - S_WK[i]) + S_tau_WK[i]))

    return S_WK, S_SP, S_xi_WK, S_xi_SP, S_K_WK, S_K_SP, S_tau_SP, S_tau_WK


# calculate Gradient：
def Recurr_gradient(X, Y, theta, i, S_WK, S_SP, S_xi_WK, S_xi_SP, S_K_WK, S_K_SP, S_tau_SP, S_tau_WK, WK_time, SP_time):
    t = X[i]
    RT = theta[0]
    xi = theta[1]
    K = theta[2]
    tau_d = theta[3]
    d_1 = theta[4]
    d_2 = theta[5]
    p = theta[6]
    p_ = theta[7]
    beta = theta[8]

    Rt_d_1 = 0
    Rt_d_2 = 0
    Rt_p = 0
    Rt_p_ = 0
    Rt_beta = 0
    I = -1
    Type = 'E'
    Start_time_I = -1
    for u in range(1, 12):
        if (SP_time[u - 1] <= t and t <= WK_time[u]):
            I = u
            Type = 'W'
            Start_time_I = SP_time[u - 1]
            break
        if (WK_time[u] <= t and t < SP_time[u]):
            I = u
            Type = 'S'
            Start_time_I = WK_time[u]
            break

    if (Type == 'W'):
        Rt_xi = S_xi_SP[I - 1]
    else:
        Rt_xi = math.exp(-(t - Start_time_I) / tau_d) * S_xi_WK[I]

    if (Type == 'W'):
        Rt_K = -(t - Start_time_I) + S_K_SP[I - 1]
    else:
        Rt_K = math.exp(-(t - Start_time_I) / tau_d) * S_K_WK[I]

    if (Type == 'W'):
        Rt_tau_d = S_tau_SP[I - 1]
    else:
        Rt_tau_d = math.exp(-(t - Start_time_I) / tau_d) * (
                    (Start_time_I - t) / tau_d / tau_d * (A_S - S_WK[I]) + S_tau_WK[I])

    if (Type == 'W'):
        Rt = S_SP[I - 1] - K * (t - Start_time_I)
    else:
        Rt = A_S - math.exp(-(t - Start_time_I) / tau_d) * (A_S - S_WK[I])
    # Rt,Rt_A_h_0,Rt_K,Rt_tau_d
    # o=gradient(X, Y, theta, i, 'Hursh_Normal')

    Ct = math.cos(2 * PI * (t - p) / 24) + beta * math.cos(4 * PI * (t - p - p_) / 24)
    Ct_xi = 0
    Ct_K = 0
    Ct_tau_d = 0
    Ct_d_1 = 0
    Ct_d_2 = 0
    Ct_p = - (PI * math.sin((PI * (p - t)) / 12)) / 12 - (PI * beta * math.sin((PI * (p + p_ - t)) / 6)) / 6
    Ct_p_ = -(PI * beta * math.sin((PI * (p + p_ - t)) / 6)) / 6
    Ct_beta = math.cos((PI * (p + p_ - t)) / 6)

    Pt = Rt * (RT * 1 - d_2 * Ct) + (d_1 + d_2) * Ct
    Pt_RT = Rt
    Pt_A_h_0 = (RT * 1 - d_2 * Ct) * Rt_xi
    Pt_K = (RT * 1 - d_2 * Ct) * Rt_K
    Pt_tau_d = (RT * 1 - d_2 * Ct) * Rt_tau_d
    Pt_d_1 = Ct
    Pt_d_2 = (Ct - (Ct * Rt))
    Pt_p = (d_1 + d_2 * (1 - Rt)) * Ct_p
    Pt_p_ = (d_1 + d_2 * (1 - Rt)) * Ct_p_
    Pt_beta = (d_1 + d_2 * (1 - Rt)) * Ct_beta

    return np.array([Pt_RT , Pt_A_h_0, Pt_K, Pt_tau_d, Pt_d_1, Pt_d_2, Pt_p, Pt_p_, Pt_beta]) * (Pt - Y[i])


# calculate loss for test sets：
def Recurr_computeCost_forTestSet(X, Y, theta, i, WK_time, SP_time, S_SP, S_WK):
    RT = theta[0]
    A_h_0 = theta[1]
    K = theta[2]
    tau_d = theta[3]
    d_1 = theta[4]
    d_2 = theta[5]
    p = theta[6]
    p_ = theta[7]
    beta = theta[8]

    t = X[i]

    for u in range(1, 12):
        if (SP_time[u - 1] <= t and t <= WK_time[u]):
            I = u
            Type = 'W'
            Start_time_I = SP_time[u - 1]
            break
        if (WK_time[u] <= t and t < SP_time[u]):
            I = u
            Type = 'S'
            Start_time_I = WK_time[u]
            break

    if (Type == 'W'):
        Rt = S_SP[I - 1] - K * (t - Start_time_I)
    else:
        Rt = A_S - math.exp(-(t - Start_time_I) / tau_d) * (A_S - S_WK[I])

    Ct = math.cos(2 * PI * (t - p) / 24) + beta * math.cos(4 * PI * (t - p - p_) / 24)

    Pt = (Rt * (RT * 1 - d_2 * Ct) + (d_1 + d_2) * Ct)

    if (Pt >= 0) and (Pt <= 1000):
        inner = 0
    if (Pt < 0):
        Pt = -1500
        inner = np.power((Pt - 500), 2)
    if (Pt > 1000):
        Pt = 1500
        inner = np.power((Pt - 500), 2)
    #print(X,X.shape,i,Type,Ct,Rt,Start_time_I,I,S_SP[I - 1] ,Pt)
    return (float(inner / 2))


# calculate Gradient for test sets：
def Recurr_gradient_forTestSet(X, Y, theta, i, S_WK, S_SP, S_xi_WK, S_xi_SP, S_K_WK, S_K_SP, S_tau_SP, S_tau_WK,
                               WK_time, SP_time):
    t = X[i]
    RT = theta[0]
    xi = theta[1]
    K = theta[2]
    tau_d = theta[3]
    d_1 = theta[4]
    d_2 = theta[5]
    p = theta[6]
    p_ = theta[7]
    beta = theta[8]

    Rt_d_1 = 0
    Rt_d_2 = 0
    Rt_p = 0
    Rt_p_ = 0
    Rt_beta = 0
    I = -1
    Type = 'E'
    Start_time_I = -1
    for u in range(1, 12):
        if (SP_time[u - 1] <= t and t <= WK_time[u]):
            I = u
            Type = 'W'
            Start_time_I = SP_time[u - 1]
            break
        if (WK_time[u] <= t and t < SP_time[u]):
            I = u
            Type = 'S'
            Start_time_I = WK_time[u]
            break

    if (Type == 'W'):
        Rt_xi = S_xi_SP[I - 1]
    else:
        Rt_xi = math.exp(-(t - Start_time_I) / tau_d) * S_xi_WK[I]

    if (Type == 'W'):
        Rt_K = -(t - Start_time_I) + S_K_SP[I - 1]
    else:
        Rt_K = math.exp(-(t - Start_time_I) / tau_d) * S_K_WK[I]

    if (Type == 'W'):
        Rt_tau_d = S_tau_SP[I - 1]
    else:
        Rt_tau_d = math.exp(-(t - Start_time_I) / tau_d) * (
                (Start_time_I - t) / tau_d / tau_d * (A_S - S_WK[I]) + S_tau_WK[I])

    if (Type == 'W'):
        Rt = S_SP[I - 1] - K * (t - Start_time_I)
    else:
        Rt = A_S - math.exp(-(t - Start_time_I) / tau_d) * (A_S - S_WK[I])
    # Rt,Rt_A_h_0,Rt_K,Rt_tau_d
    # o=gradient(X, Y, theta, i, 'Hursh_Normal')

    Ct = math.cos(2 * PI * (t - p) / 24) + beta * math.cos(4 * PI * (t - p - p_) / 24)
    Ct_xi = 0
    Ct_K = 0
    Ct_tau_d = 0
    Ct_d_1 = 0
    Ct_d_2 = 0
    Ct_p = - (PI * math.sin((PI * (p - t)) / 12)) / 12 - (PI * beta * math.sin((PI * (p + p_ - t)) / 6)) / 6
    Ct_p_ = -(PI * beta * math.sin((PI * (p + p_ - t)) / 6)) / 6
    Ct_beta = math.cos((PI * (p + p_ - t)) / 6)

    Pt = Rt * (RT * 1 - d_2 * Ct) + (d_1 + d_2) * Ct
    Pt_RT = Rt
    Pt_A_h_0 = (RT * 1 - d_2 * Ct) * Rt_xi
    Pt_K = (RT * 1 - d_2 * Ct) * Rt_K
    Pt_tau_d = (RT * 1 - d_2 * Ct) * Rt_tau_d
    Pt_d_1 = Ct
    Pt_d_2 = (Ct - (Ct * Rt))
    Pt_p = (d_1 + d_2 * (1 - Rt)) * Ct_p
    Pt_p_ = (d_1 + d_2 * (1 - Rt)) * Ct_p_
    Pt_beta = (d_1 + d_2 * (1 - Rt)) * Ct_beta

    Pt = Pt
    if (Pt >= 0) and (Pt <= 1000):
        inner = 0
    if (Pt < 0):
        Pt = -1500
        inner = (Pt - 500)
    if (Pt > 1000):
        Pt = 1500
        inner = (Pt - 500)
    return np.array([Pt_RT, Pt_A_h_0, Pt_K, Pt_tau_d, Pt_d_1, Pt_d_2, Pt_p, Pt_p_, Pt_beta]) * inner
    # return np.array([Pt_A_h_0, Pt_a_s, Pt_K, Pt_tau_d, Pt_d_2, Pt_p,Pt_p_, Pt_beta]) * 100 * (Pt * 100 - Y[i])



# Adam：
def Adam_L2Regular_withLRateSche(X_train, Y_train, X_test, Y_test, theta, alpha, epoch):
    m = np.zeros(theta.shape)
    v = np.zeros(theta.shape)
    cost = np.zeros(len(X_train) + len(X_test))
    avg_cost = np.zeros(epoch)
    belta1 = 0.9
    belta2 = 0.999
    Belta1 = 0.9
    Belta2 = 0.999
    epsilon = 1e-8

    # ,SP_time
    theta_curve=[]
    for k in range(1, epoch):
        S_WK, S_SP, S_xi_WK, S_xi_SP, S_K_WK, S_K_SP, S_tau_SP, S_tau_WK = calculated_periods_end_homestatic_level(
            theta, WK_len, SP_len)

        #print(S_SP)
        diff = np.zeros(theta.shape)
        for i in range(len(X_train)):
            cost[i] = Recurr_computeCost(X_train, Y_train, theta, i, S_WK, S_SP)
            diff = diff + Recurr_gradient(X_train, Y_train, theta, i, S_WK, S_SP, S_xi_WK, S_xi_SP, S_K_WK, S_K_SP,
                                          S_tau_SP, S_tau_WK, WK_time, SP_time)

        #print(S_SP)
        for i in range(len(X_test)):
            cost[i] = Recurr_computeCost_forTestSet(X_test, Y_test, theta, i, WK_time, SP_time, S_SP, S_WK)
            diff = diff + Recurr_gradient_forTestSet(X_test, Y_test, theta, i, S_WK, S_SP, S_xi_WK, S_xi_SP, S_K_WK,
                                                     S_K_SP, S_tau_SP, S_tau_WK, WK_time, SP_time)
        g = diff + theta * Lambda
        m = belta1 * m + (1 - belta1) * g
        v = belta2 * v + (1 - belta2) * g ** 2
        m_ = m / (1 - Belta1)
        v_ = v / (1 - Belta2)
        Belta1 = Belta1 * belta1
        Belta2 = Belta2 * belta2

        # 学习率余弦退火CosineAnnealing
        learning_rate = 0.5 * learning_rate_base * (
                1 + np.cos(np.pi * (k - warmup_steps) / float(epoch_global)))
        # 线性增长的实现
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * k + warmup_learning_rate
        # 只有当global_step 仍然处于warm up阶段才会使用线性增长的学习率warmup_rate，否则使用余弦退火的学习率learning_rate
        learning_rate = np.where(k < warmup_steps, warmup_rate, learning_rate)

        theta = theta - learning_rate * (alpha * m_ / (np.sqrt(v_) + epsilon))

        avg_cost[k] = np.average(cost)

        theta_curve.append(theta)

    return theta, avg_cost, theta_curve






Tbegin=8
Activity_Len=[19,4,20,4,20,4,20,4,20,4,20,4,20,4,16,8,16,8,16,8,16,8]
Activity_Len=np.array(Activity_Len)
PVT_measured_x_all=[[8,9.596836865,11.83240848,13.74861271,15.98418432,18.06007225,19.97627649,21.57311335,23.80868496,31.95255297,34.0284409,36.10432882,38.18021674,42.33199259,42.81104365,44.88693158,46.64345213,56.22447332,58.61972861,60.21656548,62.45213709,64.36834133,66.44422925,68.67980086,70.43632141,72.67189302,80.65607735,82.57228158,84.64816951,86.72405743,88.95962904,91.03551697,92.9517212,94.86792544,105.2473651,109.3991409,111.6347125,113.2315494,115.467121,117.3833252,119.6188968,121.2157337,129.5192854,131.5951733,133.6710613,135.4275818,137.8228371,139.5793577,141.6552456,143.8908172,144.2101846,153.9508894,156.0267774,158.1026653,159.0607674,160.1785532,163.851278,164.1706454,168.1627375,170.2386255,178.0631261,180.4583814,182.3745856,184.2907899,186.3666778,188.6022494,202.8140975,205.0496691,206.9658733,208.5627102,210.9579655,212.7144861,226.9263342,228.6828547,230.9184263,233.1539979,235.0702022,237.1460901,251.5176219,253.4338261,255.509714,257.2662346,259.6614899],
[8,9.904566467,11.80913293,13.87241327,14.34855489,16.41183523,18.3164017,22.12553463,23.71267335,32.28322245,34.34650279,35.93364151,37.99692185,39.74277445,41.96476866,44.028049,46.09132934,46.56747095,56.56644491,58.3122975,60.37557784,62.43885818,64.50213852,66.40670498,68.31127145,70.53326566,72.596546,80.53223961,82.59551995,84.65880029,86.56336676,87.03950838,88.94407484,91.00735518,94.81648812,95.76877135,104.8154621,108.624595,110.5291615,112.4337279,114.814436,116.7190025,118.9409967,120.8455632,128.9399706,131.003251,133.0665313,135.2885255,136.7169504,139.0976585,140.8435111,142.7480775,144.8113579,152.9057654,154.9690457,157.032326,158.7781786,160.841459,162.7460254,164.9680196,166.8725861,169.0945803,177.1889878,179.0935543,181.3155485,183.5375427,185.2833953,187.3466756,200.2024993,203.8529183,206.0749126,209.7253316,211.788612,225.9141466,227.9774269,229.8819934,231.7865599,233.8498402,235.9131205,248.6102303,250.5147968,253.2129326,255.276213,257.9743488],
[8,10.0684224,12.27473964,14.06737239,16.27368962,18.34211202,19.99684995,22.06527235,24.27158958,32.40738438,34.61370161,36.68212401,38.47475676,40.54317917,42.19791709,44.2663395,46.61055155,48.67897396,56.81476875,58.88319115,61.08950839,62.74424631,64.95056354,66.60530147,68.8116187,70.8800411,72.81056868,80.67057382,82.87689105,84.94531345,86.7379462,88.94426344,91.01268584,93.21900307,94.873741,96.9421634,105.4916427,109.3526978,111.5590151,113.213753,113.7653323,117.4884926,119.2811254,120.7979685,129.0716581,131.553765,133.3463977,135.4148201,137.6211374,139.4137701,141.4821925,143.2748253,145.3432477,156.0990442,157.7537821,158.1674666,161.2011528,163.8211545,165.0622079,167.1306303,169.750632,178.3001113,180.0927441,182.1611665,183.8159044,186.0222216,187.6769596,201.0527578,203.9485491,206.0169715,208.2232888,210.2917112,212.2222388,226.7011956,228.3559335,230.5622507,232.6306731,234.6990956,236.4917283,250.5570007,252.7633179,254.4180558,256.624373,258.6927954],
[8,10.20628649,12.27468007,14.34307366,16.54936014,18.34196792,20.4103615,22.47875508,24.68504157,32.40704428,34.47543787,36.68172436,38.33643922,40.54272571,42.05954767,43.02479801,45.0931916,46.88579937,56.26251695,58.88248215,60.67508992,62.74348351,64.39819837,66.60448486,68.81077135,70.87916493,73.08545142,80.80745413,82.87584772,84.9442413,87.15052779,88.80524266,91.01152915,93.07992273,94.8725305,96.94092408,105.6281771,109.3512856,111.9712508,113.2122869,115.4185734,117.486967,119.6932535,121.4858613,129.4836498,131.6899363,133.7583299,135.5509376,137.6193312,139.6877248,141.6182255,143.6866191,145.7550127,153.8906941,156.0969806,157.7516954,158.9927316,159.820089,163.8189833,164.232662,167.9557704,170.1620569,177.8840596,179.9524532,182.1587397,184.2271333,186.4334198,188.0881347,202.2911039,204.3594975,206.565784,208.2204989,210.4267854,212.4951789,226.5602553,228.7665418,230.4212567,232.6275432,234.282258,236.4885445,250.9672996,252.7599074,254.8283009,257.0345874,258.8271952],
[8,10.20628649,11.86100136,14.06728784,14.48096656,16.54936014,18.75564663,22.61664799,24.27136286,32.40704428,34.47543787,36.68172436,37.92276051,40.12904699,42.19744058,44.40372707,46.47212065,47.02369227,56.67619566,58.88248215,60.67508992,62.74348351,64.81187709,66.60448486,68.67287845,70.87916493,73.08545142,81.08323995,82.87584772,84.9442413,86.73684907,87.15052779,89.21892137,91.42520786,95.14831631,95.97567375,105.0766055,108.9376069,111.0060005,112.7986082,114.8670018,117.0732883,119.2795748,121.3479684,129.4836498,131.1383647,133.2067582,135.2751518,137.2056525,139.2740461,141.0666539,143.1350474,145.203441,153.4770154,155.5454089,157.2001238,159.4064103,161.1990181,163.2674117,165.4736981,167.5420917,169.7483782,177.332488,179.4008816,181.6071681,183.8134546,185.8818482,187.6744559,200.4984962,204.3594975,206.4278911,210.2888925,211.9436073,226.5602553,228.2149702,230.4212567,232.0759715,234.282258,236.3506516,248.7610131,250.9672996,253.5872648,255.6556584,258.1377307]]

PVT_measured_y_all=[[192.2794118,188.6029412,199.894958,196.4810924,188.6029412,203.5714286,200.4201681,207.7731092,208.0357143,191.7542017,201.9957983,203.0462185,198.0567227,192.5420168,198.5819328,198.5819328,196.7436975,229.8319328,201.4705882,217.2268908,213.0252101,206.9852941,244.5378151,224.3172269,221.6911765,278.1512605,238.7605042,238.7605042,229.0441176,223.5294118,217.7521008,232.1953782,212.5,288.9180672,220.6407563,204.0966387,221.9537815,230.3571429,240.0735294,240.8613445,254.7794118,245.5882353,222.7415966,241.3865546,256.092437,241.1239496,236.9222689,279.2016807,217.4894958,222.2163866,251.1029412,251.8907563,237.447479,224.0546218,217.7521008,257.4054622,240.5987395,202.5210084,248.4768908,289.1806723,248.4768908,225.105042,234.8214286,205.4096639,261.0819328,212.762605,198.0567227,226.4180672,222.7415966,230.8823529,217.4894958,216.1764706,224.5798319,227.2058824,244.8004202,228.5189076,241.3865546,222.7415966,218.2773109,208.0357143,180.7247899,213.0252101,212.5],
[309.3023256,296.124031,351.9379845,301.1627907,308.5271318,295.3488372,329.8449612,290.6976744,332.9457364,315.503876,332.1705426,332.1705426,338.372093,356.9767442,325.9689922,316.6666667,319.379845,327.9069767,329.4573643,330.620155,345.3488372,324.4186047,318.2170543,325.1937984,328.6821705,320.9302326,346.5116279,367.4418605,322.8682171,312.4031008,341.4728682,303.4883721,291.8604651,293.7984496,293.0232558,368.2170543,310.8527132,370.1550388,329.0697674,321.3178295,335.6589147,313.5658915,320.5426357,386.0465116,358.9147287,322.0930233,402.7131783,325.1937984,325.5813953,346.8992248,311.2403101,356.9767442,403.875969,358.5271318,348.4496124,351.9379845,310.0775194,298.0620155,339.9224806,296.8992248,309.3023256,339.9224806,343.7984496,343.7984496,365.1162791,300.3875969,317.4418605,307.751938,315.503876,294.5736434,284.1085271,310.0775194,330.620155,293.7984496,275.1937984,287.2093023,267.4418605,247.6744186,271.3178295,300.3875969,282.5581395,292.248062,297.6744186,272.0930233],
[218.8571429,234.8571429,240.5714286,220.5714286,231.4285714,238.2857143,245.7142857,244.5714286,249.1428571,236.5714286,222.2857143,238.2857143,238.2857143,249.1428571,232.5714286,238.8571429,245.1428571,228,258.2857143,244,259.4285714,247.4285714,231.4285714,245.1428571,220.5714286,245.7142857,256.5714286,231.4285714,256.5714286,250.8571429,244,244,256,238.2857143,243.4285714,300.5714286,298.2857143,346.2857143,259.4285714,249.7142857,233.1428571,238.2857143,257.7142857,305.7142857,266.8571429,263.4285714,316.5714286,261.7142857,248.5714286,256,252.5714286,250.8571429,354.8571429,259.4285714,252.5714286,250.8571429,264.5714286,266.8571429,231.4285714,234.8571429,278.8571429,272,278.8571429,256,238.2857143,272,254.2857143,228,240,247.4285714,249.1428571,252.5714286,256.5714286,274.2857143,238.8571429,245.1428571,240,231.4285714,229.7142857,252.5714286,244,236.5714286,240,238.2857143],
[220.0573066,218.3381089,220.6303725,224.0687679,217.1919771,217.1919771,215.4727794,237.8223496,227.5071633,222.3495702,224.0687679,222.3495702,230.9455587,236.1031519,222.3495702,246.991404,259.5988539,261.3180516,229.7994269,238.3954155,233.2378223,233.2378223,226.3610315,214.8997135,230.9455587,222.3495702,233.2378223,222.3495702,225.7879656,214.8997135,224.0687679,213.1805158,224.0687679,217.1919771,224.0687679,337.5358166,238.3954155,256.1604585,228.0802292,233.2378223,214.8997135,225.7879656,225.7879656,277.3638968,234.9570201,238.3954155,220.6303725,193.6962751,217.1919771,222.9226361,220.0573066,230.9455587,260.1719198,236.1031519,246.991404,224.0687679,251.0028653,238.3954155,231.5186246,234.9570201,227.5071633,261.8911175,255.5873926,236.1031519,238.3954155,225.7879656,214.8997135,225.7879656,225.7879656,209.7421203,211.4613181,220.0573066,213.7535817,225.7879656,222.3495702,225.7879656,211.4613181,213.7535817,232.6647564,229.226361,204.5845272,208.0229226,218.3381089,215.4727794,211.4613181],
[308.4745763,296.6101695,351.9774011,301.1299435,307.9096045,296.6101695,329.3785311,290.960452,332.7683616,315.819209,331.0734463,332.7683616,338.4180791,355.3672316,325.9887006,317.5141243,318.6440678,327.6836158,329.3785311,331.0734463,345.1977401,324.2937853,316.9491525,324.2937853,328.2485876,320.9039548,346.8926554,366.1016949,322.5988701,311.8644068,340.1129944,303.3898305,290.960452,294.9152542,293.220339,368.3615819,310.7344633,369.4915254,329.3785311,320.9039548,335.0282486,313.559322,320.9039548,385.3107345,357.6271186,322.5988701,401.1299435,324.8587571,325.9887006,346.8926554,311.8644068,355.9322034,403.3898305,357.0621469,348.5875706,351.9774011,310.1694915,298.3050847,340.1129944,296.6101695,308.4745763,340.1129944,343.5028249,343.5028249,364.4067797,299.4350282,317.5141243,308.4745763,315.2542373,294.9152542,284.180791,310.7344633,329.3785311,294.3502825,275.1412429,287.5706215,268.3615819,247.4576271,271.7514124,299.4350282,281.920904,292.6553672,298.3050847,271.7514124]]



print(Activity_Len)
WorS_Len=['W','S','W','S','W','S','W','S']
t0 = Tbegin

WK_len=[0,Activity_Len[0],Activity_Len[2],Activity_Len[4],Activity_Len[6],Activity_Len[8],Activity_Len[10],Activity_Len[12],Activity_Len[14],Activity_Len[16],Activity_Len[18],Activity_Len[20]]
SP_len=[0,Activity_Len[1],Activity_Len[3],Activity_Len[5],Activity_Len[7],Activity_Len[9],Activity_Len[11],Activity_Len[13],Activity_Len[15],Activity_Len[17],Activity_Len[19],Activity_Len[21]]

WK_time=[t0-8]
SP_time=[t0]
for i in range (1,12):
    WK_time.append(SP_time[i-1]+WK_len[i])
    SP_time.append(WK_time[i]+SP_len[i])


# 初始化学习率、迭代轮次和参数theta：
alpha = 0.01
epoch = 150

#Adam参数
# Base learning rate after warmup.
learning_rate_base = 2
warmup_learning_rate = 1
# Compute the number of warmup batches.
warmup_steps = 20
# 初始化学习率、迭代轮次和参数theta：
epoch_global = 400
Lambda = 0.001 * 1.0 / 2


modelType = 'Hursh_Normal'
AlgorithmType = 'Adam'#'RMSprop'


RT=300 #S: 100,500,20
d_1=25 #Z: 0,50,10
d_2=25 #E: 0,50,10
p=12 #phi:0,24,3
p_=6 #phi':0,12,3
A_h_0=0.5 #0,1,0.2
beta=0.5 #0,1,0.2
K=30/R_c #0,0.1,0.02
tau_d=2.5 #0,5,1



ParameterResults_start=[]
AccuracyResults=[]
PVT_measured_list=[]

SearchParamterName="RuppData-CSD-"
SubList={0:11,1:12,2:3,3:5,4:122}

AlgorithmType = 'Adam'

for SubNum in [0]:#0, 1, 2, 3
    SubID=SubList[SubNum]
    PVT_measured_x = np.array(PVT_measured_x_all[SubNum])
    PVT_measured_y_noise=np.array(PVT_measured_y_all[SubNum])

    for inputN in np.arange(9, len(PVT_measured_y_noise), 1):
        PVT_measured_x_train=PVT_measured_x[0:inputN]
        PVT_measured_x_test=PVT_measured_x[inputN:len(PVT_measured_x)]
        PVT_measured_y_train=PVT_measured_y_noise[0:inputN]
        PVT_measured_y_test = PVT_measured_y_noise[inputN:len(PVT_measured_x)]

        print(SubID, inputN, AlgorithmType, RT,d_1, d_2, p, p_, A_h_0,  beta, K, tau_d)

        AbsoluteError_Min_train = 1000000000
        A_h_0 = 0.9
        RT = 180
        d_1 = 10  # 0.09
        d_2 = 20  # 0.07
        beta = 0.5  # 0.1
        K = -0.1  # 30/R_c
        tau_d = 2  # 4.2
        p = 18
        p_ = 3

        tic = timer()
        for RT in np.arange(200, 300, 40):
            for d_1 in np.arange(0, 50, 25):
                for d_2 in np.arange(0, 100, 50):
                    for p in np.arange(0, 24, 3):
                        for p_ in np.arange(0, 12, 3):
                                    print(SubID,inputN,AlgorithmType,RT,A_h_0, K, tau_d, d_1, d_2,  p, p_, beta)

                                    #theta = np.array([RT, A_h_0, K, tau_d, d_1, d_2,  p, p_, beta])

                                    AlgorithmType = 'Adam_L2Regular_withLRateSche'
                                    theta = np.array([RT, A_h_0, K, tau_d, d_1, d_2, p, p_, beta])
                                    theta_start=copy.copy(theta)
                                    theta_Adam_withLRateSchedule, avg_cost, theta_curve = Adam_L2Regular_withLRateSche(PVT_measured_x_train,
                                                                                                PVT_measured_y_train,
                                                                                                PVT_measured_x_test,
                                                                                                PVT_measured_y_test,
                                                                                                theta,
                                                                                                alpha, epoch)

                                    S_WK, S_SP, S_xi_WK, S_xi_SP, S_K_WK, S_K_SP, S_tau_SP, S_tau_WK = calculated_periods_end_homestatic_level(
                                        theta_Adam_withLRateSchedule, WK_len, SP_len)
                                    PVT_y_predict_train = fx(PVT_measured_x_train, theta_Adam_withLRateSchedule, S_WK, S_SP)
                                    AbsoluteError_train = PVT_measured_y_train - PVT_y_predict_train
                                    RelativeError_train = (PVT_measured_y_train - PVT_y_predict_train) / PVT_measured_y_train
                                    AbsoluteError_train = np.sqrt(sum([x ** 2 for x in AbsoluteError_train]) / len(AbsoluteError_train))
                                    RelativeError_train = np.sqrt(sum([x ** 2 for x in RelativeError_train]) / len(RelativeError_train))

                                    if len(PVT_measured_y_test) > 0:
                                        PVT_y_predict_test = fx(PVT_measured_x_test, theta_Adam_withLRateSchedule, S_WK, S_SP)
                                        AbsoluteError_test = PVT_measured_y_test - PVT_y_predict_test
                                        AbsoluteError_test = np.sqrt(
                                            sum([x ** 2 for x in AbsoluteError_test]) / len(AbsoluteError_test))
                                    else:
                                        AbsoluteError_test = 1000000000000
                                        RelativeError_test = 1000000000000
                                    
                                    if (AbsoluteError_train < AbsoluteError_Min_train):
                                        theta_start_AbsoluteError_best = copy.copy(theta_start)
                                        AbsoluteError_Min_train = AbsoluteError_train
                                        theta_est_AbsoluteError_best = copy.copy(theta_Adam_withLRateSchedule)
    

        S_WK, S_SP, S_xi_WK, S_xi_SP, S_K_WK, S_K_SP, S_tau_SP, S_tau_WK = calculated_periods_end_homestatic_level(
            theta_est_AbsoluteError_best, WK_len, SP_len)

        DT = 0.1
        PVT_measured_y_predict_plot = []
        PVT_measured_y_measured_plot = []
        T_plot = []
        for z in range(len(PVT_measured_x_train) - 1):
            for t in np.arange(PVT_measured_x_train[z], PVT_measured_x_train[z + 1], DT):
                T_plot.append(t)
                PVT_y_predict = fx([t], theta_est_AbsoluteError_best, S_WK, S_SP)[0]
                PVT_measured_y_predict_plot.append(PVT_y_predict)
        t = PVT_measured_x_train[-1]
        T_plot.append(t)
        PVT_y_predict = fx([t], theta_est_AbsoluteError_best, S_WK, S_SP)[0]
        PVT_measured_y_predict_plot.append(PVT_y_predict)

        plt.scatter(PVT_measured_x_train, PVT_measured_y_train, color='r')
        plt.scatter(T_plot, PVT_measured_y_predict_plot, color='b')
        plt.show()