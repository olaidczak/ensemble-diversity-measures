import numpy as np
import pandas as pd
import math 

def table_scores(y1, y2, y, treshold):
    """
    Function coumputes a matrix, that shows how similarly two classifiers vote.
    input: 
          y1 : list of class probabilites for classifier 1
          y2 : list of class probabilites for classifier 2
          y : list of true class labels
          treshold (float) : between 0 and 1, determines how to binarize the class probabilities output by the classifiers
    output:
          table_scores : matrix 
    """
    y1 = np.array(y1)
    y2 = np.array(y2)
    y = np.array(y)
    y1_bin = y1[:, 0] < treshold
    y2_bin = y2[:, 0] < treshold

    table_scores = np.zeros([2, 2])
    n = y.size

    for i in range(n):
        if y1_bin[i] == y[i]:
            if y2_bin[i] == y[i]:
                table_scores[0, 0] += 1
            else:
                table_scores[0,1] += 1
        else:
            if y2_bin[i] == y[i]:
                table_scores[1,0] += 1
            else:
                table_scores[1,1] += 1

    return table_scores

def Q_statistic(y1, y2, y, treshold):
    """
    Function computes Q statistic for two classifiers.
    input: 
          y1 : list of class probabilites for classifier 1
          y2 : list of class probabilites for classifier 2
          y : list of true class labels
          treshold (float) : between 0 and 1, determines how to binarize the class probabilities output by the classifiers
    output:
          Q_stat (float) : Q statistic
    """
    ts = table_scores(y1, y2, y, treshold)
    Q_stat = (ts[0,0]*ts[1,1] - ts[0,1]*ts[1,0])/(ts[0,0]*ts[1,1] + ts[0,1]*ts[1,0])
    return Q_stat

def corr_coef(y1, y2, y, treshold):
    """
    Function computes the correlation coefficient for two classifiers.
    input: 
          y1 : list of class probabilites for classifier 1
          y2 : list of class probabilites for classifier 2
          y : list of true class labels
          treshold (float) : between 0 and 1, determines how to binarize the class probabilities output by the classifiers
    output:
          corr_coef (float) : correlation coefficient
    """
    ts = table_scores(y1, y2, y, treshold)
    corr_coef = (ts[0,0] * ts[1,1] - ts[0,1] * ts[1,0])/math.sqrt((ts[0,0]+ts[0,1])*(ts[1,1]+ts[1,0])*(ts[0,0]+ts[1,0])*(ts[1,0]+ts[1,1]))
    return corr_coef

def dis_measure(y1, y2, y, treshold):
    """
    Function computes the disagreement measure for two classifiers.
    input: 
          y1 : list of class probabilites for classifier 1
          y2 : list of class probabilites for classifier 2
          y : list of true class labels
          treshold (float) : between 0 and 1, determines how to binarize the class probabilities output by the classifiers
    output:
          dis (float) :
    """
    ts = table_scores(y1, y2, y, treshold)
    dis = (ts[0,1] + ts[1,0])/(ts[1,1] + ts[0,0] + ts[0,1] + ts[1,0])
    return dis

def df_measure(y1, y2, y, treshold):
    """
    Function computes the double-fault measure for two classifiers.
    input: 
          y1 : list of class probabilites for classifier 1
          y2 : list of class probabilites for classifier 2
          y : list of true class labels
          treshold (float): between 0 and 1, determines how to binarize the class probabilities output by the classifiers
    output:
          df (float) : double-fault measure
    """
    ts = table_scores(y1, y2, y, treshold)
    df = (ts[1,1])/(ts[1,1] + ts[0,0] + ts[0,1] + ts[1,0])
    return df

def avg_measure(measure, y, treshold, *args):
    """
    Function computes the average pairwise measure for multiple classifiers
    input: 
          measure (string) : name of pairwise measure
          y : list of true class labels
          treshold (float) : between 0 and 1, determines how to binarize the class probabilities output by the classifiers
          *args : list of class probabilites for classifiers
    output:
          avg_measure (float) : chosen average pairwise measure
    """
    L = len(args)
    y = np.array(y)
    y_new = []
    for elem in args:
        y_new.append(elem)
    y_new = np.array(y_new)

    if measure == 'Qstat':
        stat = Q_statistic
    elif measure == 'Dis':
        stat = dis_measure
    elif measure == 'DoubleFault':
        stat = df_measure
    else:
        raise ValueError('Nonexistent measure name. Available measures: Qstat, Dis, DoubleFault.')

    total_q = 0
    pair_count = 0
    for i in range(L - 1):
        for k in range(i + 1, L):
            total_q += stat(y_new[k,:], y_new[i,:], y, treshold)
            pair_count += 1
    avg_measure = 2 * total_q / (L * (L - 1))
    return  avg_measure

def entropy_measure(y, treshold, *args):
    """
    Function computes the entropy measure for multiple classifiers
    input: 
          y : list of true class labels
          treshold (float) : between 0 and 1, determines how to binarize the class probabilities output by the classifiers
          *args : list of class probabilites for classifiers
    output:
          E (float) : entropy measure E
    """
    N = len(args)
    y = np.array(y)
    m = y.size
    y_bin = []

    for elem in args:
        y_bin.append(elem[:, 0] < treshold)
    y_bin = np.array(y_bin)
    #L = [sum(y_bin[:,i]) for i in range(m)] # ilosc modeli ktora zaglosowala 1

    correct = []
    tmp = 0
    for i in range(m):
        tmp = 0
        for j in range(N):
            if y_bin[j,i] == y[i]:
                tmp += 1
        correct.append(tmp)
    E = 0
    for elem in correct:
        E +=(1/(N - math.ceil(N/2)))*min(elem, N-elem)
    
    return (1/m)*E

def KW_variance(y, treshold, *args):
    """
    Function computes the Kohavi-Wolpert variance for multiple classifiers
    input: 
          y : list of true class labels
          treshold (float) : between 0 and 1, determines how to binarize the class probabilities output by the classifiers
          *args : list of class probabilites for classifiers
    output:
          KW_var (float) : Kohavi-Wolpert variance
    """
    L = len(args)
    y = np.array(y)
    N = y.size
    y_bin = []

    for elem in args:
        y_bin.append(elem[:, 0] < treshold)
    y_bin = np.array(y_bin)

    l = [] # l - number if clasiffiers that correctly recognized each row
    tmp = 0
    for i in range(N):
        tmp = 0
        for j in range(L):
            if y[i] == y_bin[j,i]:
                tmp += 1
        l.append(tmp)
    
    l = np.array(l)
    # prob1 = l/L
    # prob0 = (1-l)/L
    # variance = (1 - prob0^2 - prob1^2)/2
    KW_var = sum(l * (L-l))/(N*pow(L,2))    
    return KW_var

def ia_measure(y, treshold, *args):
    """
    Function computes the measurement of interrater agreement κ for multiple classifiers
    input: 
          y : list of true class labels
          treshold (float) : between 0 and 1, determines how to binarize the class probabilities output by the classifiers
          *args : list of class probabilites for classifiers
    output:
          K (float) : measurement of interrater agreement
    """
    L = len(args)
    y = np.array(y)
    N = y.size
    y_bin = []
    for elem in args:
        y_bin.append(elem[:, 0] < treshold)
    y_bin = np.array(y_bin)
    p_hat = sum(sum(y_bin))/(N*L)

    l = [] # l - number if clasiffiers that correctly recognized each row
    for i in range(N):
        tmp = 0
        for j in range(L):
            if y[i] == y_bin[j,i]:
                tmp += 1
        l.append(tmp)
    l = np.array(l)

    K = 1 - sum(l * (L-l))/(pow(L,2)*N*(L-1)*p_hat*(1-p_hat))
    return K
