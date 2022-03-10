# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:17:12 2018

@author: Vijay
"""
from scipy import integrate
from scipy.optimize import fsolve
from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,1,100)
alp = 2
bet = 2
y = beta.pdf(x,alp,bet)
k = 10
#q = np.sort(np.random.rand(1,k))
qinit = np.linspace(0,1,num = k,endpoint = True).reshape(1,k)
q = np.zeros((1,k))
q_true = np.zeros((1,k))
for b in range(k):
    q[0,b] = qinit[0,b]
    q_true[0,b] = qinit[0,b]
plt.plot(x,y)
plt.scatter(q,-0.1*np.ones((1,k)), color = 'green')
plt.ion()
q[0,0] = 0
q[0,k-1] = 1
P = np.eye(k)
def myIntFunc(v,l):
    return 2*(l-v)*beta.pdf(v,alp,bet)
#
    
def myFunc(l,xpm1,xpp1):
    y = integrate.quad(myIntFunc,xpm1,l,args=(l,)) - (xpp1-l)**2*(beta.pdf(l,alp,bet))
    return y
itr = 1
while itr <50:
    plt.clf()
    plt.plot(x,y)
    plt.scatter(qinit,0*np.ones((1,k)), color = 'green')
    plt.scatter(q_true,-0.1*np.ones((1,k)), color = 'orange', marker = 'D')
    plt.scatter(q,np.zeros((1,k)), color = 'orange')
    #plt.title('Iteration # = ' + str(itr))
    plt.grid()
    plt.pause(0.5)
    P1 = np.eye(k)
    P2 = np.eye(k)
    #plt.show()
    #time.sleep(1)
    for a in range(2,k-1,2): # Loop for even points
        xpp1 = q[0,a+1]
        xpm1 = q[0,a-1]
        ypp1 = beta.pdf(xpp1,alp,bet)
        ypm1 = beta.pdf(xpm1,alp,bet)
        #
        #myFunc = @(l) quad(@(v)2*(l-v).*(-6*v.*(v-1)),xpm1,l) - (xpp1-l)^2*(-6*l.*(l-1));
        q_true[0,a] = fsolve(myFunc,args=(xpm1,xpp1),x0 = xpp1/2+xpm1/2);
        #
        m = (ypp1-ypm1)/(xpp1 - xpm1)
        c = (ypm1*xpp1 - ypp1*xpm1)/(xpp1-xpm1)
        r = np.roots([-2/3*m, 2*m*xpp1, -(m*(xpp1**2+xpm1**2)+2*c*(xpm1-xpp1)),2/3*m*xpm1**3+c*(xpm1**2-xpp1**2)])
        #print(r)
        #print(str(xpm1) + ","+ str(xpp1))
        for b in r:
            if np.isreal(b) :
                if (b > xpm1) and (b <= xpp1):
                    q[0,a] = np.real(b)
                    #print(str(a)+" change "+str(q[0,a]) + "," + str(qinit[0,a]))
                    break
        thet = (q[0,a]-xpp1)/(xpm1-xpp1)
        P1[a,a-1] = thet
        P1[a,a+1] = 1- thet
        P1[a,a] = 0
    #
    for a in range(1,k-1,2): # Loop for odd points
        xpp1 = q[0,a+1]
        xpm1 = q[0,a-1]
        ypp1 = beta.pdf(xpp1,alp,bet)
        ypm1 = beta.pdf(xpm1,alp,bet)
        #
        #myFunc = @(l) quad(@(v)2*(l-v).*(-6*v.*(v-1)),xpm1,l) - (xpp1-l)^2*(-6*l.*(l-1));
        q_true[0,a] = fsolve(myFunc,args=(xpm1,xpp1),x0 = xpp1/2+xpm1/2);
        #
        m = (ypp1-ypm1)/(xpp1 - xpm1)
        c = (ypm1*xpp1 - ypp1*xpm1)/(xpp1-xpm1)
        r = np.roots([-2/3*m, 2*m*xpp1, -(m*(xpp1**2+xpm1**2)+2*c*(xpm1-xpp1)),2/3*m*xpm1**3+c*(xpm1**2-xpp1**2)])
        #print(r)
        #print(str(xpm1) + ","+ str(xpp1))
        for b in r:
            if np.isreal(b) :
                if (b > xpm1) and (b <= xpp1):
                    q[0,a] = np.real(b)
                    #print(str(a)+" change "+str(q[0,a]) + "," + str(qinit[0,a]))
                    break
        thet = (q[0,a]-xpp1)/(xpm1-xpp1)
        P2[a,a-1] = thet
        P2[a,a+1] = 1- thet
        P2[a,a] = 0
    itr += 1
    #P = P.dot(P1).dot(P2)
    P = P1.dot(P2)
    [uu,vv] = np.linalg.eig(P)
    print("Eigenvalues" + str(np.round(uu,3)))
plt.scatter(q,np.zeros((1,k)), color = 'black')
#plt.scatter(q_init,-0.1*np.ones((1,k)), color = 'green')
plt.show()