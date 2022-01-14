import numpy as np
import matplotlib.pyplot as plt


'''
plotting function
'''
def add_line(epsilons, data, lstyle, mkr, lbl, color = 'b', ci = 95):
    med = np.median(data, axis = 1)

    lower_ci = []
    upper_ci = []
    p=data.shape[0]
    for i in range(p):
        lower_ci.append(np.percentile(data[i,:],50-ci/2))
        upper_ci.append(np.percentile(data[i,:],50+ci/2))
    plt.fill_between(epsilons, lower_ci, upper_ci, alpha=0.25, color = color)
    # plt.rcParams["text.usetex"] =True
    l = plt.plot(epsilons, med, color = color, linestyle=lstyle, marker=mkr,  linewidth=.5, label = lbl)
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
REK functions
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
    z_lsqsol = b - A @ lsqsol
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


        # if not np.allclose(StM @ y, Stc, 1e-12, 1e-12):
        #     print('constraint not satisfied')
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
    m = 2000
    n = 100

    n_iters = 10000 #number of iterations

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
    zxs_saprek = []


    epsilons = []
    for j in range(10):
        np.random.seed(0)
        zxs_saprek_it = []
        epsilons.append(10**(j-5))
        for i in range(n_trials):
            
            trial = sapreksolver(x,b,A,b,n_iters,epsilons[j])
            # zxs_saprek_it.append(trial[1] + epsilons[j]* trial[0])
            zxs_saprek_it.append(trial[0])
            print('Epsilon '+str(epsilons[j])+' Trial '+str(i)+' Finished')
        zxs_saprek.append(np.vstack(zxs_saprek_it).T)
    zxs_saprek = np.dstack(zxs_saprek).T   


    #make the plotss
    LINESTYLES = ["-", "--", ":", "-."]
    MARKERS = ['D', 'o', 'X', '*', '<', 'd', 'S', '>', 's', 'v']
    COLORS = ['b','k','c','m','y']

    for i in range(4):
        add_line(epsilons, zxs_saprek[:,:,(i+1)*2500-1], LINESTYLES[(i+1) % 4], MARKERS[i+1], 'Iteration k='+str((i+1)*2500), COLORS[i])
    plt.plot(10000*[10e-30], linestyle = '-', color = 'k', label = 'REK Iteration 2500')
    plt.yscale('log')
    plt.xscale('log')
    # plt.legend()
    plt.xlabel('Epsilon')
    # plt.rcParams["mathtext.fontset"] = "cm"
    # plt.ylabel('$||\mathbf{z}^k - \mathbf{z}^*||_2^2 + \epsilon||\mathbf{x}^k - \mathbf{x}^*||_2^2$')
    plt.ylabel('$||\mathbf{x}^k - \mathbf{x}^*||^2$')
    plt.legend()
    plt.savefig('./figs/gaussian_x_epsilons'+str(m)+'x'+str(n)+'.pdf')
    plt.close()
    



    
    #coherent example
    A = np.random.uniform(0, 1, size = (m,n))

    n_trials = 50

    #run the trials
    #(this is über bad coding)
    zxs_saprek = []


    epsilons = []
    for j in range(10):
        np.random.seed(0)
        zxs_saprek_it = []
        epsilons.append(10**(j-5))
        for i in range(n_trials):
            
            trial = sapreksolver(x,b,A,b,n_iters,epsilons[j])
            # zxs_saprek_it.append(trial[1] + epsilons[j]* trial[0])
            zxs_saprek_it.append(trial[0])
            print('Epsilon '+str(epsilons[j])+' Trial '+str(i)+' Finished')
        zxs_saprek.append(np.vstack(zxs_saprek_it).T)

    zxs_saprek = np.dstack(zxs_saprek).T   


    #make the plotss
    LINESTYLES = ["-", "--", ":", "-."]
    MARKERS = ['D', 'o', 'X', '*', '<', 'd', 'S', '>', 's', 'v']
    COLORS = ['b','k','c','m','y']

    line_objects = []

    for i in range(4):
        line_objects.append(add_line(epsilons, zxs_saprek[:,:,(i+1)*2500-1], LINESTYLES[(i+1) % 4], MARKERS[i+1], 'Iteration k='+str((i+1)*2500), COLORS[i]))
    ll = plt.plot(10000*[10e-30], linestyle = '-', color = 'k', label = 'REK Iteration 2500')
    plt.yscale('log')
    plt.xscale('log')
    # plt.legend()
    plt.xlabel('Epsilon')
    # plt.rcParams["mathtext.fontset"] = "cm"
    # plt.ylabel('$||\mathbf{z}^k - \mathbf{z}^*||_2^2 + \epsilon||\mathbf{x}^k - \mathbf{x}^*||_2^2$')
    plt.ylabel('$||\mathbf{x}^k - \mathbf{x}^*||^2$')
    plt.savefig('./figs/coherent_x_epsilons'+str(m)+'x'+str(n)+'.pdf')
    plt.close()

    lines = [l[0] for l in line_objects]
    lines.append(ll[0])

    labels = []
    for i in range(4):
        num_its = (i+1)*2500
        labels.append('Iteration k='+str(num_its))
    
    labels.append('REK Iteration 2500')


    import pylab
    fig = pylab.figure()
    figlegend = pylab.figure(figsize=(3,2))
    ax = fig.add_subplot(111)
    figlegend.legend(lines, labels)
    figlegend.savefig('./figs/legend_eps.pdf')
    plt.close()
    
