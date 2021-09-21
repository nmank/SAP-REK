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

def calc_lambda(sigma, eps):

    y1 = 1/(m + 1/eps)

    y20 = (((m-2+2/eps)*sigma**2 + 1)**2/4 - (sigma**4/eps)*((m+1/eps) - sigma**2))**(.5)
    y2 = ( ( m-2 + 2/eps)*sigma**2 + 1 )/(2*(m+1/eps) ) - (1/(m+1/eps))*y20

    lmbda = np.min([y1,y2])
    print(np.argmin([y1,y2]))

    return lmbda




#################################s#########

#build examples 200x10 matrix

if __name__ == "__main__":
    m=200
    n=10

    A_g = np.random.normal(0, 1, size = (m,n))
    A_c = np.random.uniform(0, 1, size = (m,n))

    s_g = np.min(np.linalg.svd(A_g)[1])
    s_c = np.min(np.linalg.svd(A_c)[1])
    
    eps = []
    lambdas_g = []
    lambdas_c = []
    for j in range(45):
        eps.append(10**((j/5-5)))
        print(eps)
        lambdas_g.append(calc_lambda(s_g, eps[j]))
        lambdas_c.append(calc_lambda(s_c, eps[j]))
        
    plt.plot(eps, lambdas_g, label = 'Gaussian')
    plt.plot(eps, lambdas_c, label = 'Coherent', linestyle = 'dashed')

    plt.xlabel('Epsilon')
    plt.xscale('log')
    plt.ylabel('Lambda')
    plt.legend()
    plt.savefig('./figs/sigma_test.pdf')
    plt.close()

    #separate plots
    plt.plot(eps, lambdas_g, label = 'Gaussian')
    
    plt.xlabel('Epsilon')
    plt.xscale('log')
    plt.ylabel('Lambda')
    # plt.legend()
    plt.savefig('./figs/sigma_test_gaussian.pdf')
    plt.close()

    plt.plot(eps, lambdas_c, label = 'Coherent')
    plt.xlabel('Epsilon')
    plt.xscale('log')
    plt.ylabel('Lambda')
    # plt.legend()
    plt.savefig('./figs/sigma_test_coherent.pdf')
    plt.close()

    