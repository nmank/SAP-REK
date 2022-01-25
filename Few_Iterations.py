import numpy as np
import matplotlib.pyplot as plt


'''
plotting function
'''
def add_line(data, lstyle, mkr, lbl, mkevery, color = 'b', ci = 95):
    med = np.median(data, axis = 0)

    lower_ci = []
    upper_ci = []
    for i in range(data.shape[1]):
        lower_ci.append(np.percentile(data[:,i],50-ci/2))
        upper_ci.append(np.percentile(data[:,i],50+ci/2))
    plt.fill_between(list(np.arange(len(med))), lower_ci, upper_ci, alpha=0.25, color = color)
    # plt.rcParams["text.usetex"] =True
    l = plt.plot(med, color = color, linestyle=lstyle, marker=mkr, markevery = mkevery, linewidth=.5, label = lbl)
    return l
##########################


'''
RK function
inputs: n_iters number of iterations
outputs: squared error at each iteration
'''
def rksolver(x,A,b,n_iters):
    m,n = A.shape
    lsqsol = np.linalg.lstsq(A,b, rcond=None)[0] #least squares solution
    y1 = np.zeros(n_iters)
    for k in range(n_iters):
        i = np.random.randint(m)
        x = x - ((A[[i], :] @ x - b[i])/(np.linalg.norm(A[i, :])**2)) * A[[i],:].T
        y1[k] = np.linalg.norm(x-lsqsol)**2
    return y1

'''
REK function
inputs: n_iters number of iterations
outputs: squared error at each iteration
'''
def reksolver(x,z,A,b,n_iters):
    m,n = A.shape
    lsqsol = np.linalg.lstsq(A,b, rcond=None)[0]
    M = np.block([[A.T, np.zeros((n,n))], [np.eye(m), A]]) 
    c = np.vstack([np.zeros((n,1)), b])
    # z_lsqsol = np.linalg.lstsq(M,c, rcond=None)[0][:m]
    # z_lsqsol = (np.eye(m) -  A @ np.linalg.pinv(A)) @ b
    z_lsqsol = b - A@ lsqsol
    y1 = np.zeros(n_iters)
    z1 = np.zeros(n_iters)
    for k in range(n_iters):
        i = np.random.randint(m)
        j = np.random.randint(n)
        z = z - ((A[:,[j]].T @ z)/(np.linalg.norm(A[:,j])**2)) * A[:,[j]] #z update
        x = x -((A[[i],:]@x-b[i,0]+z[i,0])/(np.linalg.norm(A[i,:])**2))*A[[i],:].T #x update
        y1[k] = np.linalg.norm(x-lsqsol)**2
        z1[k] = np.linalg.norm(z-z_lsqsol)**2
    return y1, z1

'''
SAP-REK function
inputs: n_iters number of iterations
        eps is the epsilon in the B matrix
outputs: squared error at each iteration
'''
def sapreksolver(x,z,A,b,n_iters,eps):
    m,n = A.shape
    lsqsol = np.linalg.lstsq(A,b, rcond=None)[0]
    x_err = np.empty(n_iters)
    z_err = np.empty(n_iters)
    Binv = np.block([
        [eps*np.eye(m),       np.zeros((m,n))], 
        [np.zeros((n,m)), np.eye(n)]
    ]) #B inverse matrix
    # z_lsqsol = np.linalg.lstsq(M,c, rcond=None)[0][:m]
    # z_lsqsol = (np.eye(m) -  A @ np.linalg.pinv(A)) @ b
    z_lsqsol = b - A @ lsqsol
    y = np.vstack([z, x])
    for k in range(n_iters):
        i = np.random.randint(m)
        j = np.random.randint(n)
        StM = np.block([
            [A[:,[j]].T,       np.zeros((1,n))],
            [np.eye(m)[[i],:], A[[i],:]]
        ])
        Stc = np.array([
            [0],
            [b[i]]
        ])
        y = y - Binv @ StM.T @ np.linalg.pinv(StM @ Binv @ StM.T) @ (StM @ y - Stc) #SAP update
        z = y[:m]
        x = y[m:n+m]
        if not np.allclose(StM @ y, Stc, 1e-12, 1e-12):
            print('constraint not satisfied')
        x_err[k] = np.linalg.norm(x-lsqsol)**2
        z_err[k] = np.linalg.norm(z-z_lsqsol)**2
        if k > 0 and (z_err[k] + x_err[k]*eps) > (z_err[k-1] + x_err[k-1]*eps):
            print('not monotonicly decreasing!')
            print('iteration '+ str(k))
            print((z_err[k] + x_err[k]*eps),  (z_err[k-1] + x_err[k-1]*eps))
    return x_err, z_err



##########################################

#build examples 200x10 matrix

if __name__ == "__main__":
    #matrix dimensions
    m = 10000
    n = 100

    n_iters = 500 #number of iterations

    b = np.random.rand(m,1)
    x = np.random.rand(n,1)
    z = np.random.rand(m,1)
    zg = np.random.rand(n,1)
    x1 = np.array(list(np.arange(0, n_iters)))
    A = np.random.normal(0, 1, size = (m,n))
    M = np.block([[A.T, np.zeros((n,n))], [np.eye(m), A]])
    c = np.vstack([np.zeros((n,1)), b])
    y = np.vstack([z, x])

    #50 trials per method
    n_trials = 50





    #run the trials
    #(this is über bad coding)
    xs_rek = []
    zs_rek = []

    xs_saprek = []
    zs_saprek = []
    np.random.seed(0)
    for i in range(n_trials):
        trial = reksolver(x,b,A,b,n_iters)
        xs_rek.append(trial[0])
        zs_rek.append(trial[1])

    xs_rek = np.vstack(xs_rek)
    zs_rek = np.vstack(zs_rek)

    #make the plots
    LINESTYLES = ["-", "--", ":", "-."]
    MARKERS = ['D', 'o', 'X', '*', '<', 'd', 'S', '>', 's', 'v']
    COLORS = ['b','k','c','m','y']
    # COLORS = ['y','m','c','k','b']
    # MARKERS = ['D','d','<','*','X','o']
    # LINESTYLES = ['-.',':','--','-']


    for j in range(5):
        np.random.seed(0)
        xs_saprek_it = []
        zs_saprek_it = []
        for i in range(n_trials):
            eps = 10**(-j)
            trial = sapreksolver(x,b,A,b,n_iters,eps)
            xs_saprek_it.append(trial[0])
            zs_saprek_it.append(trial[1])
            print('Epsilon '+str(eps)+' Trial '+str(i)+' Finished')
        xs_saprek.append(np.vstack(xs_saprek_it))
        zs_saprek.append(np.vstack(zs_saprek_it))



    for i in range(5):
        eps = 10**(-i)
        plt.figure('x_err')
        add_line(xs_saprek[i], LINESTYLES[(i+1) % 4], MARKERS[i+1], 'SAP-REK($\epsilon = {}$)'.format(eps), 50, COLORS[i])
        plt.figure('z_err')
        add_line(zs_saprek[i], LINESTYLES[(i+1) % 4], MARKERS[i+1], '$\epsilon = {}$'.format(eps), 50, COLORS[i])
        plt.figure('xz_err')
        add_line(zs_saprek[i]+ eps* xs_saprek[i], LINESTYLES[(i+1) % 4], MARKERS[i+1], '$\epsilon = {}$'.format(eps), 50, COLORS[i])

    plt.figure('x_err')
    add_line(xs_rek, LINESTYLES[0], MARKERS[0], 'REK', 50, 'g')
    plt.legend()
    plt.figure('z_err')
    add_line(zs_rek, LINESTYLES[0], MARKERS[0], 'REK', 50, 'g')
    plt.figure('xz_err')
    add_line(zs_rek, LINESTYLES[0], MARKERS[0], 'REK', 50, 'g')

    
    for figname in ['x_err','z_err','xz_err']:
        plt.figure(figname)
        plt.yscale('log')
        plt.xlabel('Iteration $k$')
    

    plt.figure('x_err')
    # plt.ylim(10**(-14),10)
    plt.ylim(10**(-2),10**2)
    plt.ylabel('$||\mathbf{x}^k - \mathbf{x}^*||^2$')
    plt.savefig('./figs/gaussian_REK_few_itr'+str(m)+'x'+str(n)+'.pdf')

    # plt.figure('z_err')
    # plt.ylim(10**(-14),10)
    # plt.ylabel('$||\mathbf{z}^k - \mathbf{z}^*||^2$')
    # plt.savefig('./figs/gaussian_REK_zs.pdf')

        
    # plt.figure('xz_err')
    # plt.ylim(10**(-14),10)
    # plt.ylabel('$||\mathbf{z}^k - \mathbf{z}^*||^2 + \epsilon||\mathbf{x}^k - \mathbf{x}^*||^2$')
    # plt.savefig('./figs/gaussian_REK_converges.pdf')

    plt.close()
    plt.close()
    plt.close()



    
    #coherent example
    n_iters = 1200
    A = np.random.uniform(0, 1, size = (m,n))

    #run the trials
    #(this is über bad coding)
    xs_rek = []
    zs_rek = []

    xs_saprek = []
    zs_saprek = []
    np.random.seed(0)
    for i in range(n_trials):
        trial = reksolver(x,b,A,b,n_iters)
        xs_rek.append(trial[0])
        zs_rek.append(trial[1])

    xs_rek = np.vstack(xs_rek)
    zs_rek = np.vstack(zs_rek)


    for j in range(5):
        np.random.seed(0)
        xs_saprek_it = []
        zs_saprek_it = []
        for i in range(n_trials):
            eps = 10**(-j)
            trial = sapreksolver(x,b,A,b,n_iters,eps)
            xs_saprek_it.append(trial[0])
            zs_saprek_it.append(trial[1])
            print('Epsilon '+str(eps)+' Trial '+str(i)+' Finished')
        xs_saprek.append(np.vstack(xs_saprek_it))
        zs_saprek.append(np.vstack(zs_saprek_it))

    line_objects = []



    for i in range(5):
        eps = 10**(-i)
        plt.figure('x_err')
        line_objects.append(add_line(xs_saprek[i], LINESTYLES[(i+1) % 4], MARKERS[i+1], '$\epsilon = {}$'.format(eps), 80, COLORS[i]))
        plt.figure('z_err')
        add_line(zs_saprek[i], LINESTYLES[(i+1) % 4], MARKERS[i+1], '$\epsilon = {}$'.format(eps), 80, COLORS[i])
        plt.figure('xz_err')
        add_line(zs_saprek[i]+ eps* xs_saprek[i], LINESTYLES[(i+1) % 4], MARKERS[i+1], '$\epsilon = {}$'.format(eps), 80, COLORS[i])

    plt.figure('x_err')
    line_objects.append(add_line(xs_rek, LINESTYLES[0], MARKERS[0], 'REK', 80, 'g'))
    plt.figure('z_err')
    add_line(zs_rek, LINESTYLES[0], MARKERS[0], 'REK', 80, 'g')
    plt.figure('xz_err')
    add_line(zs_rek, LINESTYLES[0], MARKERS[0], 'REK', 80, 'g')

    for figname in ['x_err','z_err','xz_err']:
        plt.figure(figname)
        plt.yscale('log')
        plt.xlabel('Iteration $k$')

    plt.figure('x_err')
    # plt.ylim(10**(-14),10)
    plt.ylim(10**(-2),10**2)
    plt.ylabel('$||\mathbf{x}^k - \mathbf{x}^*||^2$')
    plt.savefig('./figs/coherent_REK_few_itr'+str(m)+'x'+str(n)+'.pdf')

    # plt.figure('z_err')
    # plt.ylim(10**(-14),10)
    # plt.ylabel('$||\mathbf{z}^k - \mathbf{z}^*||^2$')
    # plt.savefig('./figs/coherent_REK_zs.pdf')

        
    # plt.figure('xz_err')
    # plt.ylim(10**(-14),10)
    # plt.ylabel('$||\mathbf{z}^k - \mathbf{z}^*||^2 + \epsilon||\mathbf{x}^k - \mathbf{x}^*||^2$')
    # plt.savefig('./figs/coherent_REK_converges.pdf')

 

    lines = [line_objects[i][0] for i in range(6)]

    labels = []
    
    for i in range(5):
        eps = 10**(-i)
        labels.append('SAP-REK($\epsilon = {}$)'.format(eps))
    labels.append('REK')


    import pylab
    fig = pylab.figure()
    figlegend = pylab.figure(figsize=(3,2))
    ax = fig.add_subplot(111)
    figlegend.legend(lines, labels, 'center')
    figlegend.savefig('./figs/legend_few_it.pdf')
    plt.close()