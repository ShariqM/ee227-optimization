import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pdb
from math import exp
from math import sqrt
from numpy.linalg import norm

#x0 = np.array([-1001.2, -1001.2])
steps = 1000
grad_limit = 10e-3

def f_a(x):
    return x[0] ** 2 + 5 * x[1] ** 2 + x[0] - 5 * x[1]

def df_a(x):
    df = np.zeros(2)
    df[0] = 2  * x[0] + 1
    df[1] = 10 * x[1] - 5
    return df

def f_b(x):
    return x[0]**2 + 5*x[0]*x[1] + 100*x[1]**2 - x[0]+4*x[1]

def df_b(x):
    df = np.zeros(2)
    df[0] = 2*x[0] + 5*x[1] - 1
    df[1] = 5*x[0] + 200*x[1] + 4
    return df

def f_c(x):
    return exp(x[0] + 3*x[1] - 0.1) + exp(x[0] - 3*x[1] - 0.1) + exp(-x[0] - 0.1)

def df_c(x):
    df = np.zeros(2)
    df[0] = exp(x[0] + 3*x[1] - 0.1) + exp(x[0] - 3*x[1] - 0.1) - exp(-x[0] - 0.1)
    df[1] = 3*exp(x[0] + 3*x[1] - 0.1) - 3*exp(x[0] - 3*x[1] - 0.1)
    return df

def f_d(x):
    return 100*(x[1] - x[0]**2)**2 + (1-x[0])**2

def df_d(x):
    df = np.zeros(2)
    df[0] = 400*x[0]**3 -400*x[0]*x[1] + 2*x[0] - 2
    df[1] = 200*(x[1] - x[0]**2)
    return df

def f_e(x):
    return 100*(x[1] - x[0]**2)**2 + (1-x[0])**2 + 50 * (x[2] - x[1]**2)**2

def df_e(x):
    df = np.zeros(3)
    df[0] = 400*x[0]**3 -400*x[0]*x[1] + 2*x[0] - 2
    df[1] = 200*(x[1] - x[0]**2) - 4 * (x[2] - x[1]**2) * x[1]
    df[2] = 2 * (x[2] - x[1]**2)
    return df

def df(x):
    if x < 1:
        return 50*x
    if 1 <= x and x <= 2:
        return 2*x + 48
    if x > 2:
        return 50*x - 48
    raise Exception("Unhandled case")

def plot_results(name, x, fx):
    plt.title("BackTracking Line Search (f_%s), x*=[%.2f, %.2f]" % (name, x[0], x[1]), fontdict={'fontsize':20})
    plt.xlabel("iteration #", fontdict={'fontsize':18})
    plt.ylabel("f(x)", fontdict={'fontsize':18})
    plt.plot(fx)
    plt.show()

def plot_contours(name, f, x_rec, fx):
    plt.title("BackTracking Line Search (f_%s)" % (name), fontdict={'fontsize':20})
    plt.xlabel("x_1", fontdict={'fontsize':18})
    plt.ylabel("x_2", fontdict={'fontsize':18})

    delta = 0.025
    x = np.arange(-1.5, 1.5, delta)
    y = np.arange(-1.5, 1.5, delta)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i][j] = f([x[i], y[j]])
    plt.contour(X,Y,Z)
    plt.colorbar()

    sz = x_rec.shape[1]
    for i in range(sz):
        if i % (x_rec.shape[1] / 10) == 0:
            print i
            c = float(i)/float(x_rec.shape[1]), 0.0, float(x_rec.shape[1]-i)/float(x_rec.shape[1]) #R,G,B
            plt.scatter(x_rec[0,i], x_rec[1,i], color=c, s=20)

    plt.show()

def back_method(fname, f, df):
    gamma = 0.4
    beta = 0.99
    t = 1.

    x = x0
    fx = np.zeros(steps)
    fx[0] = f(x)
    x_rec = np.zeros((x0.shape[0], steps))
    x_rec[:,0] = x

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
        x_rec[:,i] = x

    fx[i:steps] = fx[i-1]
    print "\ti = %d, x = %s, fx=%.3f, t=%.3f, L = %.2f" % (i, x, fx[i-1], t, L)
    return fx, i
    ##plot_results(fname, x, fx[:i])
    #plot_contours(fname, f, x_rec[:,:i], fx[:i])

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
    m,L = get_mL(x, x + 1e-4 * df(x), df)
    t = 1./L
    fx = np.zeros(steps)
    fx[0] = f(x)
    x_rec = np.zeros((x.shape[0], steps))
    x_rec[:,0] = x
    Ls = np.zeros(steps)
    Ls[0] = L

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
        x_rec[:,i] = x
        Ls[i] = L
        print "\ti=%d, x=%s, fx=%.6f, t=%.3f, L=%.2f m=%.2f" % (i, x, fx[i], t, L, m)

    fx[i:steps] = fx[i-1]
    print "\ti=%d, x=%s, fx=%.3f, t=%.3f, L=%.2f m=%.2f" % (i, x, fx[i-1], t, L, m)
    return fx, i, Ls

    #plot_results(fname, x, fx[:i])
    #plot_contours(fname, f, x_rec[:,:i], fx[:i])

def get_rates(x, x_new, df):
    grad_diff = df(x) - df(x_new)
    param_diff = x - x_new
    L = norm(grad_diff) / norm(param_diff)
    m = np.dot(grad_diff, param_diff)/(norm(param_diff) ** 2)
    if m < 0:
        print '*** m < 0 ***'
        m = L
    kappa = L/m
    alpha = 4./((sqrt(L) + sqrt(m)) ** 2)
    beta = (sqrt(kappa) - 1) / (sqrt(kappa) + 1)
    return L, m, kappa, alpha, beta

def ball_method(fname, f, df):
    x_new = x0 - 1e-4 * df(x0)
    L, m, kappa, alpha, beta = get_rates(x0, x_new, df)

    x_curr = x0
    x_past = x_curr
    fx = np.zeros(steps)
    fx[0] = f(x_curr)
    alphas = np.zeros(steps)
    betas  = np.zeros(steps)
    alphas[0] = alpha
    betas[0]  = beta

    x_rec = np.zeros((x0.shape[0], steps))

    for i in range(1, steps):
        df_x = df(x_curr)
        if norm(df_x) <= grad_limit:
            print 'Norm limit, i=%d' % i
            break

        x = x_curr - alpha * df(x_curr) + beta * (x_curr - x_past)
        x_past = x_curr
        x_curr = x
        fx[i] = f(x)

        #x_rec[i] = x_curr
        L, m, kappa, alpha, beta = get_rates(x_past, x_curr, df)
        alphas[i] = alpha
        betas[i] = beta

    fx[i:steps] = fx[i-1]
    print "\ti=%d, x=%s, fx=%.3f, a=%.3f, L=%.2f m=%.2f" % (i, x, fx[i-1], alpha, L, m)
    return fx, i, alphas, betas
    #plot_results(name, x, fx, x_upper)


def theta_update(theta_prev):
    return 1./2 * (-theta_prev**2 + sqrt(theta_prev**4 + 4*theta_prev**2))

def ball_method_2(fname, f, df):
    print 'Ball 2'
    x_new = x0 - 1e-4 * df(x0)
    L, g, g, g, g = get_rates(x0, x_new, df)

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

        x = x_curr - alpha * df(x_curr) + beta * (x_curr - x_past)
        x_past = x_curr
        x_curr = x
        fx[i] = f(x)

        #x_rec[i] = x_curr
        alphas[i] = alpha
        betas[i] = beta
        L, g, g, g, g = get_rates(x_past, x_curr, df)
        print "\ti=%d, x=%s, fx=%.3f, a=%.3f, L=%.2f, T=%.2f" % (i, x, fx[i], alpha, L, theta[i])

    return fx, i, alphas, betas
    #plot_results(name, x, fx, x_upper)


def nesterov_method(fname, f, df):
    print 'Nesterov'
    x_new = x0 - 1e-4 * df(x0)
    L, g, g, g, g = get_rates(x0, x_new, df)

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
            break

        alphas[i] = alpha
        betas[i] = beta
        L, g, g, g, g = get_rates(x_past, x_curr, df)
        print "\ti=%d, x=%s, fx=%.3f, a=%.3f, L=%.2f T=%.2f" % (i, x, fx[i], alpha, L, theta[i])

    return fx, i, alphas, betas

def optimize_help(grad_type, func_name):
    if func_name == 'a':
        return grad_type(func_name, f_a, df_a)
    if func_name == 'b':
        return grad_type(func_name, f_d, df_d)
    if func_name == 'c':
        return grad_type(func_name, f_c, df_c)
    if func_name == 'd':
        return grad_type(func_name, f_d, df_d)
    if func_name == 'e':
        return grad_type(func_name, f_e, df_e)
    raise Exception("Unknown func")

def run(func_name):
    print 'Optimizing %s' % func_name
    fx_grad, end_grad, Ls = optimize_help(grad_method, func_name)
    fx_back, end_back = optimize_help(back_method, func_name)
    fx_ball, end_ball, alphas, betas = optimize_help(ball_method, func_name)
    fx_nest, end_nest, alphas, betas = optimize_help(nesterov_method, func_name)
    #fx_bal2, end_bal2, alphas, betas = optimize_help(ball_method_2, func_name)

    end = min([end_grad, end_back, end_ball])
    eps = max(fx_grad) * 1e-1

    plt.title("Convergence for f_%s" % (func_name), fontdict={'fontsize':20})
    lw=6
    plt.plot(fx_grad[:end], label='Gradient method (iter=%d)' % end_grad, ls='--', lw=lw)
    plt.plot(fx_back[:end], label='Backtracking method (iter=%d)' % end_back, ls=':', lw=lw)
    #plt.plot(fx_ball[:end], label='Heavy Ball method (iter=%d)' % end_ball, ls='-.', lw=lw)
    plt.plot(fx_nest[:end], label='Nesterov method (iter=%d)' % end_nest, ls='-.', lw=lw)
    #plt.plot(fx_bal2[:end], label='Ball method (iter=%d)' % end_bal2, ls='-.', lw=lw)

    plt.xlabel("iteration #", fontdict={'fontsize':18})
    plt.ylabel("f(x)", fontdict={'fontsize':18})
    plt.axis([0, end - 1, min(fx_grad) - eps, max(fx_grad) + eps])
    plt.legend()
    plt.show()

    plt.title("L over time f_%s" % (func_name), fontdict={'fontsize':20})
    plt.xlabel("iteration #", fontdict={'fontsize':18})
    plt.ylabel("L_emp", fontdict={'fontsize':18})
    plt.plot(Ls[:end])
    plt.legend()
    plt.show()

#x0 = np.array([-1.2, 1, 4])
#for func in ('e'):
x0 = np.array([-1.2, 1])
#for func in ('a', 'b', 'c'):
#for func in ('a', 'b'):
for func in ('c'):
    run(func)
