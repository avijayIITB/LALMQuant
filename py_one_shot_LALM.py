# -*- coding: utf-8 -*-
"""
Description : One shot LALM
Created on Thu Aug 22 17:28:50 2019

@author: Vijay
"""

import numpy as np
import energyusage
from scipy.stats import beta
import matplotlib.pyplot as plt
import scipy.integrate as integrate
#
def py_MISE(alp,bet,k,q,f):
    """
    Estimates the Mean Integrated Square Error (MISE) of the piecewise linear approximation of the density 
    """
    MISE = 0
    for aa in range(k-1):
        m = (f[aa+1] - f[aa])/(q[aa+1]-q[aa])
        c = (q[aa+1]*f[aa]-q[aa]*f[aa+1])/(q[aa+1]-q[aa])
        err = lambda  x : (m*x+c - beta.pdf(x,alp,bet))**2
        MISE += integrate.quad(err,q[aa],q[aa+1])[0]
    return MISE
#
def py_1D_local_PLD_Approx(binCount, N, mu, qp,qn):
    """
    Function to compute the local PLD approximation
    binCount -> Data points inside the bin
    N -> Total Number of Data points
    mu -> (sample) mean if the data
    qp -> left quantization level
    qn -> right quantization level    
    """
    C = np.array([[(qn-qp)/2, (qn-qp)/2], [qn**2/3-qn*qp/6-qp**2/6 - mu*qn/2+mu*qp/2, -(qp**2/3-qn*qp/6-qn**2/6 + mu*qn/2 - mu*qp/2)]]) 
    b = np.array([binCount/N,0])
    f = np.linalg.solve(C,b)
#    print(np.shape(f))
    return f

def py_1D_method1_PLD_Approx(data_pts, quant_pts, quant_levels):
    """
    Function to compute the PLD approximation, local probability and whole mean
    binCount -> Data points inside the bin
    data_pts -> Colloctetion of data point    
    quant_pts -> the set of quantization points
    quant_levels -> number of quantization levels
    """
    bins = np.zeros((1,quant_levels-1))
    C = np.zeros((quant_levels,quant_levels))
    b = np.zeros((quant_levels,1))
    # Bin counts
    for a in range(quant_levels-1):
        if a == quant_levels-2:
            index = (data_pts <= quant_pts[a+1]) & (data_pts >= quant_pts[a])
            bins[0,a] = np.sum(data_pts[index])
        else:
            index = (data_pts < quant_pts[a+1]) & (data_pts >= quant_pts[a])
            bins[0,a] = np.sum(data_pts[index])
        #
        b[0:quant_levels-1,0] = bins[0,:]/np.shape(data_pts)[1]
        b[quant_levels-1,0] = 1
    for a in range(quant_levels-1):
        C[a,a] = quant_pts[a+1]**2/6 - quant_pts[a]**2/3 + quant_pts[a]*quant_pts[a+1]/6
        C[a,a+1] = quant_pts[a+1]**2/3 -quant_pts[a]**2/6 - quant_pts[a]*quant_pts[a+1]/6
    for a in range(quant_levels):
        if a==1:
            prev = quant_pts[a]
            nxt = quant_pts[a+1]
        elif a == quant_levels-1:
            prev = quant_pts[a-1]
            nxt = quant_pts[a]
        else :
            prev = quant_pts[a-1]
            nxt = quant_pts[a+1]
        C[quant_levels-1,a] = (nxt - prev)/2
    f = np.linalg.solve(C,b)
    return f
#
def py_1D_method2_PLD_Approx(data_pts, quant_pts, quant_levels):
    bins = np.zeros((1,quant_levels-1))
    C = np.zeros((quant_levels,quant_levels))
    b = np.zeros((quant_levels,1))
    # Bin counts
    for a in range(quant_levels-1):
        if a == quant_levels-2:
            index = (data_pts <= quant_pts[a+1]) & (data_pts >= quant_pts[a])
            bins[0,a] = np.sum(data_pts[index])
        else:
            index = (data_pts < quant_pts[a+1]) & (data_pts >= quant_pts[a])
            bins[0,a] = np.sum(data_pts[index])
    b[0:quant_levels-1,0] = bins[0,:]/np.shape(data_pts)[1]
    b[quant_levels-1,0] = 1
    for a in range(quant_levels-1):
        C[a,a] = quant_pts[a+1]**2/6 - quant_pts[a]**2/3 + quant_pts[a]*quant_pts[a+1]/6
        C[a,a+1] = quant_pts[a+1]**2/3 -quant_pts[a]**2/6 - quant_pts[a]*quant_pts[a+1]/6
    for a in range(quant_levels):
        if a==1:
            prev = quant_pts[a]
            nxt = quant_pts[a+1]
        elif a== quant_levels-1 :
            prev = quant_pts[a-1]
            nxt = quant_pts[a]
        else:
            prev = quant_pts[a-1]
            nxt = quant_pts[a+1]
        C[quant_levels-1,a] = (nxt - prev)/2 
    f = np.linalg.solve(C,b)
    return f

#
def py_lloyd_1shot_LALM(X,alp,bet,k):
    """
    -> X is the input data stream (1-D)
    -> k is the number of quantization levels
    -> N is the total number of data points    
    """
    N = np.shape(X)[1]
    MaxIter = 1000    
    q = np.linspace(np.amin(X),np.amax(X),num=k, endpoint = True)
    q_prev = q.copy()
    f = beta.pdf(q,alp,bet)  
    #
    #--- Plots are commented ---#
    plt.plot(q_prev,f)
    plt.scatter(q,np.zeros((1,k)))
    #
#    f1 = lambda t : t*beta.pdf(t,alp,bet)
    # 
    # One shot step
    for aa in range(k-2,0,-1):
        #
        index = (X >= q[aa-1]) & (X <= q[aa+1])
        if np.sum(index.astype(int)) == 0:
            #q[aa] = (q[aa-1] + q[aa+1])/2
            continue
        mu = np.sum(X[index])/np.sum(index.astype(int))
        ff = py_1D_local_PLD_Approx(np.sum(index.astype(int)),N,mu,q[aa-1],q[aa+1])
        f[aa+1] = ff[0]
        f[aa-1] = ff[1]
    #    
    itr = 0 
    while itr < MaxIter:
        itr = itr + 1;
        #
        #--- Plots are commented ---#
        if (np.mod(itr,MaxIter/10) ==0):
            plt.clf()            
            plt.plot(q,f)        
            plt.scatter(q,np.zeros((1,k)),s=20, facecolors='None', edgecolors='y',marker ='o')
            plt.title("Iteration : [ %d ]" % (itr) )
            plt.pause(0.25)
        #
        #
        q_prev = q.copy()
        for aa in range(k-2,0,-1):
            #
            xpp1 = q[aa+1]
            xpm1 = q[aa-1]
            ypp1 = f[aa+1]
            ypm1 = f[aa-1]
            #
            m = (ypp1 - ypm1)/(xpp1 - xpm1);
            c = (ypm1*xpp1 - ypp1*xpm1)/(xpp1-xpm1)                        
            #            
            z = np.sqrt(ypp1**2 + ypp1*ypm1 + ypm1**2)
            q[aa] = (z/np.sqrt(3) - c)/m
            #
            if (aa==1)|(aa==k-2):
                z = np.sqrt(1/2*ypp1**2 + ypp1*ypm1 + 1/2*ypm1**2)
                q[aa] = (z/np.sqrt(2) - c)/m
            #
            if q[aa] > q_prev[aa]:
                m_int = (f[aa+1]-f[aa])/(q_prev[aa+1] - q_prev[aa])
                c_int = (q_prev[aa+1]*f[aa]-q_prev[aa]*f[aa+1])/(q_prev[aa+1] - q_prev[aa])
                f[aa] = m_int*q[aa] +c_int
            elif q[aa] < q_prev[aa]:
                m_int = (f[aa]-f[aa-1])/(q_prev[aa] - q_prev[aa-1])
                c_int = (q_prev[aa]*f[aa-1]-q_prev[aa-1]*f[aa])/(q_prev[aa] - q_prev[aa-1])
                f[aa] = m_int*q[aa] +c_int
            else:
                continue
    # MSE Calculation    
    fMSE = lambda  l,x : (l-x)**2*beta.pdf(x,alp,bet)
    SMSE = 0
    for aa in range(1,k-1):
        if aa == 1 :
            b_l = 0
            b_h = (q[aa]+q[aa+1])/2
        elif aa==k-2 :
            b_l = (q[aa-1]+q[aa])/2
            b_h = 1
        else :
            b_l = (q[aa-1]+q[aa])/2
            b_h = (q[aa]+q[aa+1])/2
        SMSE = SMSE + integrate.quad(lambda y : fMSE(q[aa],y),b_l,b_h)[0]    
        #
    f[f<0] = 0
    A = 0    
    ff_meth1 = py_1D_method1_PLD_Approx(X,q,k)
    ff_meth1[ff_meth1 <0] = 0
    ff_meth2 = py_1D_method2_PLD_Approx(X,q,k)    
    ff_meth2[ff_meth2 <0] = 0
    A_meth1 = 0
    A_meth2 = 0
    for aa in range(k-1):
        A += (q[aa+1] -q[aa])*(f[aa]+f[aa+1])/2
        A_meth1 += (q[aa+1] -q[aa])*(ff_meth1[aa]+ff_meth1[aa+1])/2
        A_meth2 += (q[aa+1] -q[aa])*(ff_meth2[aa]+ff_meth2[aa+1])/2
    f /=A        
    ff_meth1 /= A_meth1
    ff_meth2 /= A_meth2
    plt.clf()
    plt.plot(q,f)
    plt.plot(q,ff_meth1)
    plt.plot(q,ff_meth2)
    plt.plot(np.linspace(0,1,100,endpoint=True),beta.pdf(np.linspace(0,1,100,endpoint=True),alp,bet))       
    plt.pause(0.1)
    MISE = py_MISE(alp,bet,k,q,f)
    MISE_meth1 = py_MISE(alp,bet,k,q,ff_meth1)
    MISE_meth2 = py_MISE(alp,bet,k,q,ff_meth2)
    print("(1S-LALM) Final MSE for k = %d and N = %d points : %f;   MISE = %f  and MISE1 = %f, MISE2 = %f\n" % (k,N,SMSE,MISE,MISE_meth1,MISE_meth2))     
    return MISE,MISE_meth1, MISE_meth2, SMSE,q
# 
# ------------------------ LALM (usual) ------------------------
#
def py_lloyd_local_LALM(X,alp,bet,k):
    """
    -> X is the input data stream (1-D)
    -> k is the number of quantization levels
    -> N is the total number of data points    
    """
    N = np.shape(X)[1]
    MaxIter = 1000    
    q = np.linspace(np.amin(X),np.amax(X),num=k, endpoint = True)
    q_init = q.copy()
    f = beta.pdf(q,alp,bet)  
    """
    #--- Plots are commented ---#
    plt.plot(q_init,f)
    plt.scatter(q,np.zeros((1,k)))
    """
#    f1 = lambda t : t*beta.pdf(t,alp,bet)
    #
    itr = 0 
    while itr < MaxIter:
        itr = itr + 1;
        """
        #--- Plots are commented ---#
        if (np.mod(itr,MaxIter/10) ==0):
            plt.clf()            
            plt.plot(q,f)        
            plt.scatter(q,np.zeros((1,k)),s=20, facecolors='None', edgecolors='y',marker ='o')
            plt.title("Iteration : [ %d ]" % (itr) )
            plt.pause(0.25)
        """
        #
        for aa in range(k-2,0,-1):
            #
            index = (X >= q[aa-1]) & (X <= q[aa+1])
            if np.sum(index.astype(int)) == 0:
                #q[aa] = (q[aa-1] + q[aa+1])/2
                continue
            mu = np.sum(X[index])/np.sum(index.astype(int))
            ff = py_1D_local_PLD_Approx(np.sum(index.astype(int)),N,mu,q[aa-1],q[aa+1])
            f[aa+1] = ff[0]
            f[aa-1] = ff[1]
            xpp1 = q[aa+1];
            xpm1 = q[aa-1]
            ypp1 = ff[0]
            ypm1 = ff[1]
            #
            m = (ypp1 - ypm1)/(xpp1 - xpm1);
            c = (ypm1*xpp1 - ypp1*xpm1)/(xpp1-xpm1)                        
            #            
            z = np.sqrt(ypp1**2 + ypp1*ypm1 + ypm1**2)
            q[aa] = (z/np.sqrt(3) - c)/m;
            #
            if (aa==1)|(aa==k-2):
                z = np.sqrt(1/2*ypp1**2 + ypp1*ypm1 + 1/2*ypm1**2)
                q[aa] = (z/np.sqrt(2) - c)/m;
            
    # MSE Calculation    
    fMSE = lambda  l,x : (l-x)**2*beta.pdf(x,alp,bet)
    SMSE = 0
    for aa in range(1,k-1):
        if aa == 1 :
            b_l = 0
            b_h = (q[aa]+q[aa+1])/2
        elif aa==k-2 :
            b_l = (q[aa-1]+q[aa])/2
            b_h = 1
        else :
            b_l = (q[aa-1]+q[aa])/2
            b_h = (q[aa]+q[aa+1])/2
        SMSE = SMSE + integrate.quad(lambda y : fMSE(q[aa],y),b_l,b_h)[0]    
        #
    f[f<0] = 0
    A = 0    
    ff_meth1 = py_1D_method1_PLD_Approx(X,q,k)
    ff_meth1[ff_meth1 <0] = 0
    ff_meth2 = py_1D_method2_PLD_Approx(X,q,k)    
    ff_meth2[ff_meth2 <0] = 0
    A_meth1 = 0
    A_meth2 = 0
    for aa in range(k-1):
        A += (q[aa+1] -q[aa])*(f[aa]+f[aa+1])/2
        A_meth1 += (q[aa+1] -q[aa])*(ff_meth1[aa]+ff_meth1[aa+1])/2
        A_meth2 += (q[aa+1] -q[aa])*(ff_meth2[aa]+ff_meth2[aa+1])/2
    f /=A        
    ff_meth1 /= A_meth1
    ff_meth2 /= A_meth2
    plt.clf()
    plt.plot(q,f)
    plt.plot(q,ff_meth1)
    plt.plot(q,ff_meth2)
    plt.plot(np.linspace(0,1,100,endpoint=True),beta.pdf(np.linspace(0,1,100,endpoint=True),alp,bet))       
    plt.pause(0.1)
    MISE = py_MISE(alp,bet,k,q,f)
    MISE_meth1 = py_MISE(alp,bet,k,q,ff_meth1)
    MISE_meth2 = py_MISE(alp,bet,k,q,ff_meth2)
    print("(LALM) Final MSE for k = %d and N = %d points : %f;   MISE = %f  and MISE1 = %f, MISE2 = %f\n" % (k,N,SMSE,MISE,MISE_meth1,MISE_meth2))     
    return MISE,MISE_meth1, MISE_meth2, SMSE,q
#                 
#                 
alp = 4
bet = 2
k =10
data_points = np.random.beta(alp,bet,[1,50000])
_,_,_, S_1S_LALM, q_1S = py_lloyd_1shot_LALM(data_points,alp,bet,k)
_,_,_, S_LALM, q = py_lloyd_local_LALM(data_points,alp,bet,k)
x = np.linspace(0,1,100,endpoint=True)
plt.figure()
plt.plot(x,beta.pdf(x,alp,bet))
plt.scatter(q_1S,np.zeros((1,k)))
plt.scatter(q,0.25*np.ones((1,k)))
plt.show()