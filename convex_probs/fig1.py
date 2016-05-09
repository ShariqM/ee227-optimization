import numpy as np
import sys
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pdb
from math import exp
from math import sqrt
from numpy.linalg import norm

steps = 1000
grad_limit = 10e-3
init_dist = 1e-4

def f_a(x):
    return x[0] ** 2 + 5 * x[1] ** 2 + x[0] - 5 * x[1]

def df_a(x):
    df = np.zeros(2)
    df[0] = 2  * x[0] + 1
    df[1] = 10 * x[1] - 5
    return df

def f_b(x):
    return 100*(x[1] - x[0]**2)**2 + (1-x[0])**2

def df_b(x):
    df = np.zeros(2)
    df[0] = 400*x[0]**3 -400*x[0]*x[1] + 2*x[0] - 2
    df[1] = 200*(x[1] - x[0]**2)
    return df

def f_c(x):
    return exp(x[0] + 3*x[1] - 0.1) + exp(x[0] - 3*x[1] - 0.1) + exp(-x[0] - 0.1)

def df_c(x):
    df = np.zeros(2)
    df[0] = exp(x[0] + 3*x[1] - 0.1) + exp(x[0] - 3*x[1] - 0.1) - exp(-x[0] - 0.1)
    df[1] = 3*exp(x[0] + 3*x[1] - 0.1) - 3*exp(x[0] - 3*x[1] - 0.1)
    return df

def back_method(fname, f, df):
    gamma = 0.4
    beta = 0.8
    t = 1.

    x = x0
    fx = np.zeros(steps)
    fx[0] = f(x)

    for i in range(1, steps):
        df_x = df(x)
        if norm(df_x) <= grad_limit:
            print 'Norm limit, i=%d' % i
            break
        x_new = x - t * df_x
        while f(x_new) > f(x) - gamma*t*norm(df_x)**2:
            t = beta * t
            x_new = x - t * df_x

        L = norm(df(x) - df(x_new)) / norm(x - x_new)
        x = x_new
        fx[i] = f(x)

    fx[i:steps] = fx[i-1]
    print "\ti = %d, x = %s, fx=%.3f, t=%.3f, L = %.2f" % (i, x, fx[i-1], t, L)
    return fx, i

def get_tpss(x, x_new, df):
    grad_diff = df(x) - df(x_new)
    param_diff = x - x_new
    return np.dot(grad_diff, param_diff) / (norm(grad_diff) ** 2)

def get_mL(x, x_new, df):
    grad_diff = df(x) - df(x_new)
    param_diff = x - x_new
    L = norm(grad_diff) / norm(param_diff)
    m = np.dot(grad_diff, param_diff)/(norm(param_diff) ** 2)
    if m < 0:
        print 'm < 0 !!!!'
        m = L
    return m, L

def grad_method(fname, f, df):
    x = x0
    m,L = get_mL(x, x + init_dist * df(x), df)
    t = 1./L
    fx = np.zeros(steps)
    fx[0] = f(x)
    Ls = np.zeros(steps)
    Ls[0] = L
    print (L)

    gamma = 0.0 # Forget the past

    for i in range(1, steps):
        df_x = df(x)
        if norm(df_x) <= grad_limit:
            print 'Norm limit, i=%d' % i
            break
        x_new = x - t * df_x

        m,tL = get_mL(x, x_new, df)
        L = gamma * L + (1 - gamma) * tL
        t = 1./(L)
        x = x_new
        fx[i] = f(x)
        Ls[i] = L
        print "\ti=%d, x=%s, fx=%.6f, t=%.3f, L=%.2f m=%.2f" % (i, x, fx[i], t, L, m)

    fx[i:steps] = fx[i-1]
    print "\ti=%d, x=%s, fx=%.3f, t=%.3f, L=%.2f m=%.2f" % (i, x, fx[i-1], t, L, m)
    return fx, i, Ls

def tpss_method(fname, f, df):
    x = x0
    t = get_tpss(x, x + init_dist * df(x), df)
    fx = np.zeros(steps)
    fx[0] = f(x)
    Ts = np.zeros(steps)
    Ts[0] = t

    for i in range(1, steps):
        df_x = df(x)
        if norm(df_x) <= grad_limit:
            print 'Norm limit, i=%d' % i
            break
        x_new = x - t * df_x

        t = get_tpss(x, x_new, df)
        x = x_new
        fx[i] = f(x)
        Ts[i] = t
        print "\ti=%d, x=%s, fx=%.6f, t=%.3f" % (i, x, fx[i], t)

    fx[i:steps] = fx[i-1]
    print "\ti=%d, x=%s, fx=%.3f, t=%.3f" % (i, x, fx[i-1], t)
    return fx, i, Ts

def get_L(x, x_new, df):
    grad_diff = df(x) - df(x_new)
    param_diff = x - x_new
    return norm(grad_diff) / norm(param_diff)

def theta_update(theta_prev):
    return 1./2 * (-theta_prev**2 + sqrt(theta_prev**4 + 4*theta_prev**2))

def nesterov_method(fname, f, df):
    print 'Nesterov'
    x_new = x0 - init_dist * df(x0)
    L = get_L(x0, x_new, df)

    x_curr = x0
    x_past = x_curr
    fx = np.zeros(steps)
    fx[0] = f(x_curr)
    alphas = np.zeros(steps)
    betas  = np.zeros(steps)
    theta = np.zeros(steps)
    theta[0] = 1.

    for i in range(1, steps):
        df_x = df(x_curr)
        if norm(df_x) <= grad_limit:
            print 'Norm limit, i=%d' % i
            break
        alpha = 1./L
        theta[i] = theta_update(theta[i-1])
        beta     = theta[i] * (1./theta[i-1] - 1)

        x = x_curr - alpha * df(x_curr + beta * (x_curr - x_past)) + beta * (x_curr - x_past)
        x_past = x_curr
        x_curr = x
        try:
            fx[i] = f(x)
        except Exception as e:
            print "Exploded, x=%s" % x
            fx[i:] = fx[i-1]
            i = 999
            break

        alphas[i] = alpha
        betas[i] = beta
        L = get_L(x_past, x_curr, df)
        print "\ti=%d, x=%s, fx=%.3f, a=%.3f, L=%.2f T=%.2f" % (i, x, fx[i], alpha, L, theta[i])

    return fx, i, alphas, betas

def optimize_help(grad_type, func_name):
    if func_name == 'a':
        return grad_type(func_name, f_a, df_a)
    if func_name == 'b':
        return grad_type(func_name, f_b, df_b)
    if func_name == 'c':
        return grad_type(func_name, f_c, df_c)
    raise Exception("Unknown func")

def run(func_name):
    print 'Optimizing %s' % func_name
    fx_grad, end_grad, Ls = optimize_help(grad_method, func_name)
    fx_tpss, end_tpss, Ts = optimize_help(tpss_method, func_name)
    fx_back, end_back = optimize_help(back_method, func_name)
    fx_nest, end_nest, alphas, betas = optimize_help(nesterov_method, func_name)

    end = min([end_grad, end_back, end_nest])
    eps = max(fx_grad) * 1e-1

    tsize = 24
    asize = 22
    lsize = 18
    plt.title("Convergence for f_%s" % (func_name), fontdict={'fontsize':tsize})
    lw=6
    plt.plot(fx_grad[:end], label='Empirical Gradient method (iter=%d)' % end_grad, ls='--', lw=lw)
    plt.plot(fx_back[:end], label='Backtracking method (iter=%d)' % end_back, ls=':', lw=lw)
    plt.plot(fx_nest[:end], label='Empirical Nesterov method (iter=%d)' % end_nest, ls='-.', lw=lw)
    plt.plot(fx_tpss[:end], label='Two-Point Step Size method (iter=%d)' % end_tpss, ls='-', lw=3)

    plt.xlabel("iteration # (k)", fontdict={'fontsize':asize})
    plt.ylabel("f(x_k)", fontdict={'fontsize':asize})
    plt.axis([0, end - 1, min(fx_grad) - eps, max(fx_grad) + eps])
    plt.legend(prop={'size':lsize})
    plt.show()

    plt.title("L over time f_%s" % (func_name), fontdict={'fontsize':tsize})
    plt.xlabel("iteration #", fontdict={'fontsize':asize})
    plt.ylabel("L_k", fontdict={'fontsize':asize})
    plt.axis([0, end, 0, np.max(Ls) * 1.1])
    plt.plot(Ls[:end])
    #plt.legend(prop={'size':lsize})
    plt.show()

x0 = np.array([-1.2, 1])
for func in ('a', 'b', 'c'):
#for func in ('b', 'c'):
#for func in ('c'):
    run(func)
