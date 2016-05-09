import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from math import sqrt
import pdb

x_star = 0
x0 = 3
steps = 30

L = 50.
m = 2.
kappa = L/m

grad_limit = 10e-4

def f(x):
    if x < 1:
        return 25 * x ** 2
    if 1 <= x and x <= 2:
        return x ** 2 + 48*x - 24
    if x > 2:
        return 25 * x ** 2 - 48*x + 72
        #return 25 * x ** 4 - 48*x + 72
    raise Exception("Unhandled case")

def df(x):
    if x < 1:
        return 50*x
    if 1 <= x and x <= 2:
        return 2*x + 48
    if x > 2:
        #return 100*x**3 - 48
        return 50*x - 48
    raise Exception("Unhandled case")

def plot_results(name, x, fx, x_upper=None):
    plt.title("%s Method, x*=%.2f" % (name, x))
    plt.xlabel("iteration #", fontdict={'fontsize':18})
    plt.ylabel("f(x)", fontdict={'fontsize':18})
    plt.plot(fx, label='f(x)')
    if False or x_upper is not None:
        plt.plot(x_upper, label='upper bound')
    plt.legend()
    plt.show()


def init():
    x = x0
    fx = np.zeros(steps)
    fx[0] = f(x)
    return x, fx

def get_rates(x, x_new):
    grad_diff = df(x) - df(x_new)
    param_diff = x - x_new
    L = norm(grad_diff) / norm(param_diff)
    m = np.dot(grad_diff, param_diff)/(norm(param_diff) ** 2)
    if m < 0:
        print 'm < 0'
        m = L
    kappa = L/m
    alpha = 4./((sqrt(L) + sqrt(m)) ** 2)
    if alpha != alpha:
        pdb.set_trace()
    beta = (sqrt(kappa) - 1) / (sqrt(kappa) + 1)
    return L, m, kappa, alpha, beta

def grad_method(optimal=False):
    if optimal:
        eta = 2./(L+m)
        name = "Gradient (Optimal)"
    else:
        eta = 1./50
        name = "Gradient"

    x, fx = init()
    contraction = (kappa - 1) / (kappa + 1)

    x_upper = np.zeros(steps)
    x_upper[0] = f(x)
    for k in range(1, steps):
        df_x = df(x)
        if norm(df_x) <= grad_limit:
            print 'Norm limit, k=%d' % k
            break
        x = x - eta * df(x)
        fx[k] = f(x)
        x_upper[k] = (contraction) ** k * (f(x0) - f(x_star)) + f(x_star)
        print "\tk=%d, x=%.3f, fx=%.3f, t=%.3f" % (k, x, fx[k], eta)
    return fx[:k]

def online_grad_method():
    x, fx = init()
    x_tmp = x - 1e-4 * df(x)
    L, m, kappa, alpha, beta = get_rates(x, x_tmp)
    Ls = np.zeros(steps)
    Ls[0] = L
    eta = 2./(L+m)

    for k in range(1, steps):
        df_x = df(x)
        if norm(df_x) <= grad_limit:
            print 'Norm limit, k=%d' % k
            break
        x = x - eta * df(x)
        fx[k] = f(x)
        print "\tk=%d, x=%.3f, fx=%.3f, t=%.3f" % (k, x, fx[k], eta)
    return fx[:k], Ls[:k]

def ball_method(optimal=False):
    if optimal:
        alpha = 4./((sqrt(L) + sqrt(m)) ** 2)
        beta = (sqrt(kappa) - 1) / (sqrt(kappa) + 1)
        name = "Heavy Ball (Optimal)"
    else:
        alpha = 1./18
        beta = 4./9
        name = "Heavy Ball"
    contraction = (sqrt(kappa) - 1) / (sqrt(kappa) + 1)

    x_curr, fx = init()
    x_past = x_curr
    x_upper = np.zeros(steps)
    x_upper[0] = f(x_curr)
    for k in range(1, steps):
        x = x_curr - alpha * df(x_curr) + beta * (x_curr - x_past)
        x_past = x_curr
        x_curr = x
        fx[k] = f(x)
        x_upper[k] = (contraction) ** k * (f(x0) - f(x_star)) + f(x_star)
        print "x_curr=%.2f" % x_curr

    #plot_results(name, x, fx, x_upper)

def online_ball_method():
    x_curr, fx = init()
    x_past = x_curr
    x_tmp = x_curr - 1e-4 * df(x_curr)
    L, m, kappa, alpha, beta = get_rates(x_past, x_tmp)

    for k in range(1, steps):
        df_x = df(x_curr)
        if norm(df_x) <= grad_limit:
            print 'Norm limit, k=%d' % k
            break

        x = x_curr - alpha * df_x + beta * (x_curr - x_past)
        x_past = x_curr
        x_curr = x

        fx[k] = f(x)
        print "\tk=%d, x=%.3f, fx=%.3f, t=%.3f, L=%.2f, m=%.2f" % (k, x, fx[k], alpha, L, m)

        L, m, kappa, alpha, beta = get_rates(x_past, x_curr)

def online_nest_method():
    name = "Nesterov (Empirical)"
    x_curr, fx = init()
    x_past = x_curr
    x_tmp = x_curr - 1e-4 * df(x_curr)
    L, m, kappa, alpha, beta = get_rates(x_past, x_tmp)
    Ls = np.zeros(steps)
    Ls[0] = L

    for k in range(1, steps):
        df_x = df(x_curr)
        if norm(df_x) <= grad_limit:
            print 'Norm limit, k=%d' % k
            break

        x = x_curr - alpha * df(x_curr + beta * (x_curr - x_past)) + beta * (x_curr - x_past)
        x_past = x_curr
        x_curr = x

        fx[k] = f(x)
        print "\tk=%d, x=%.3f, fx=%.3f, t=%.3f, L=%.2f, m=%.2f" % (k, x, fx[k], alpha, L, m)

        L, m, kappa, alpha, beta = get_rates(x_past, x_curr)
        Ls[k] = L
    return fx[:k], Ls[:k]

def nest_method(optimal=False):
    if optimal:
        eta = 1./50
        beta = (sqrt(kappa) - 1) / (sqrt(kappa) + 1)
        name = "Nesterov (Optimal)"
    else:
        eta = 1./50
        beta = 2./3
        name = "Nesterov"

    contraction = (1 - (1./sqrt(kappa)))

    x_curr, fx = init()
    x_past = x_curr
    x_upper = np.zeros(steps)
    x_upper[0] = f(x_curr)

    for k in range(1, steps):
        df_x = df(x_curr)
        if norm(df_x) <= grad_limit:
            print 'Norm limit, k=%d' % k
            break

        x = x_curr - eta * df(x_curr + beta * (x_curr - x_past)) + beta * (x_curr - x_past)
        x_past = x_curr
        x_curr = x
        fx[k] = f(x)
        x_upper[k] = (contraction) ** k * (f(x0) - f(x_star)) + f(x_star)
        print "\tk=%d, x=%.3f, fx=%.3f, t=%.3f" % (k, x, fx[k], eta)

    return fx[:k]

def do_work():
    grad_e_fx, grad_e_Ls = online_grad_method()
    grad_fx = grad_method(optimal=True)

    tsize = 24
    asize = 22
    lsize = 22
    lw=6
    plt.title("Convergence for f_strong", fontdict={'fontsize':tsize})
    plt.plot(grad_e_fx, label='Gradient (Empirical) iter=%d' % grad_e_fx.shape[0], ls='--', lw=lw)
    plt.plot(grad_fx, label='Gradient iter=%d' % grad_fx.shape[0], ls=':', lw=lw)
    plt.xlabel("iteration #", fontdict={'fontsize':asize})
    plt.ylabel("f(x)", fontdict={'fontsize':asize})
    plt.legend(prop={'size':lsize})
    plt.show()


    plt.title("L over time f_strong", fontdict={'fontsize':tsize})
    plt.xlabel("iteration #", fontdict={'fontsize':asize})
    plt.ylabel("L_k", fontdict={'fontsize':asize})
    plt.plot(grad_e_Ls)
    plt.show()

    nest_e_fx, nest_e_Ls = online_nest_method()
    nest_fx = nest_method(optimal=True)

    lw=6
    plt.title("Convergence for f_strong", fontdict={'fontsize':tsize})
    plt.plot(nest_e_fx, label='Nesterov (Empirical) iter=%d' % nest_e_fx.shape[0], ls='--', lw=lw)
    plt.plot(nest_fx, label='Nesterov iter=%d' % nest_fx.shape[0], ls=':', lw=lw)
    plt.xlabel("iteration #", fontdict={'fontsize':asize})
    plt.ylabel("f(x)", fontdict={'fontsize':asize})
    plt.legend(prop={'size':lsize})
    plt.show()

    plt.title("L over time f_strong", fontdict={'fontsize':tsize})
    plt.xlabel("iteration #", fontdict={'fontsize':asize})
    plt.ylabel("L_k", fontdict={'fontsize':asize})
    plt.plot(nest_e_Ls)
    plt.show()


#ball_method(optimal=True)
#online_nest_method()
#nest_method(optimal=False)
#grad_method()
do_work()
pdb.set_trace()
ball_method()

grad_method(optimal=True)
nest_method(optimal=True)
ball_method(optimal=True)
