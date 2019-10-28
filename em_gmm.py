import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss


def gamma_ik(mu_hats, var_hats, pi_hats, data):
    """
    Compute P(Z = k | X = x_i, mu, var).

    :return nparray of shape (n, k), with P(Z = k | X = x_i, mu_k, var_k) for all n data points and k clusters.
    """
    data = np.expand_dims(data, axis=1)
    gamma_iks = ss.norm.pdf(data, mu_hats, np.sqrt(var_hats)) * pi_hats # shape: (n, k)
    gamma_iks = np.divide(gamma_iks.T, np.sum(gamma_iks,axis=1)).T # summing along axis=1 reduces from shape (n, k) to (n,). Then, element-wise divide to normalize for each data point.
    print("data:", data)   
    print("gamma_iks:", gamma_iks)   
    return gamma_iks

def e_step(mu_hats, var_hats, pi_hats, data, gamma_iks):
    """
    Compute Q, an estimate of E[log P(X, Z | mu, var, pi)] across P(Z | X) distribution.

    :param mu_hats num_k sized list of estimates for mu_k
    :param var_hats num_k sized list of estimates for var_k
    :param pi_hats num_k sized list of estimates for pi_k
    :param data n sized list of sample data points

    :return estimate of E[log P(X, Z | mu, var, pi)] across P(Z | X) distribution
    """
    data = np.expand_dims(data, axis=1)
    log_pis = np.log(pi_hats)
    # print(f"log_pis.shape: {log_pis.shape}")
    # print(mu_hats)
    log_norm_pdfs = np.log(ss.norm.pdf(data, mu_hats, np.sqrt(var_hats)))
    # print(f"log_norm_pdfs.shape: {log_norm_pdfs.shape}")
    log_pi_norm_pdfs = log_pis + log_norm_pdfs
    # print(f"log_pi_norm_pdfs.shape: {log_pi_norm_pdfs.shape}")
    # gamma_iks = gamma_ik(mu_hats, var_hats, pi_hats, data)
    # print(f"gamma_iks.shape: {gamma_iks.shape}")
    return np.sum(log_pi_norm_pdfs * gamma_iks)

def m_step(mu_hats, data, q_est, gamma_iks):
    """
    Maximize mu_k, var_k, and pi_k for all k. 
    :param mu_hats num_k sized list of estimates for mu_k
    :param data n sized list of sample data points
    :param q_est scalar estimate of 
    E[log P(X, Z | mu, var, pi)] across P(Z | X) distribution.
    :param gamma_iks P(Z = k | X_i) for all X_i, k.

    :return updated mu_hats, var_hats, and pi_hats for all k.
    """
    eps = 0.0001
    gamma_ks = np.sum(gamma_iks, axis=0) # shape is (k,)
    # print(gamma_iks)
    print("gamma_ks", gamma_ks.shape, gamma_ks)
    updated_pi_hats = gamma_ks/len(data)
    print("updated_pi_hats", updated_pi_hats.shape, updated_pi_hats)
    updated_mu_hats = np.matmul(gamma_iks.T, data)/gamma_ks
    print("updated_mu_hats", updated_mu_hats.shape, updated_mu_hats)
    #updated_var_hats = np.square(np.sum(gamma_iks * np.square(np.expand_dims(data, axis=0).T - updated_mu_hats), axis=0)/gamma_ks) + eps
    expanded_data = np.repeat(data.reshape(len(data), 1), len(updated_mu_hats), axis=1)
    # print(expanded_data)
    exp_mu = np.repeat(updated_mu_hats.reshape(1, len(updated_mu_hats)), len(data), axis=0)
    # print(exp_mu)
    differences = expanded_data - exp_mu
    # print(differences)
    squared_differences = np.square(differences)
    # print("sq diffs", squared_differences)
    # print("gik", gamma_iks)
    weighted_diffs = np.multiply(gamma_iks, squared_differences)
    # print("weighted diffs", weighted_diffs)
    sums_of_diffs = np.sum(weighted_diffs, axis=0)
    print(sums_of_diffs)
    print(gamma_ks)
    updated_var_hats = sums_of_diffs/gamma_ks + eps
    # print("updated_var_hats:", updated_var_hats)

    print("updated_var_hats", updated_var_hats.shape, updated_var_hats)

    return updated_mu_hats, updated_var_hats, updated_pi_hats

def get_true_params(num_k):
    """
    Randomly generates the true parameters (mu, variance, and pi)_k
    
    :param k the number of Gaussian to mix.
    :return (mus, variances, pis), where each element is a list of k values 
    (e.g. mus is list of mu for each of the k mixtures).
    """
    np.random.seed(0)
    return np.array([5, 15]), np.square(np.array([1, 1])), np.array([0.5, 0.5])

    # mus = np.random.randint(5, size=num_k)
    # stop = 15
    # mus = np.arange(start=0, stop=stop, step=stop/num_k)
    # variances = np.random.randint(1, 4, size=num_k)
    # pis = np.random.random(num_k)
    # pis /= pis.sum()

    # assert(sum(pis) == 1)

    # return mus, variances, pis

def gen_data(n, mus, variances, pis):
    """
    Generates n data points from Gaussian mixture model 
    defined by mus, variances, and pis.

    :return nparray of length n of points sampled from GMM.
    """
    # return np.array([4,6, 14, 16])
    num_k = len(pis)
    z = np.random.choice(num_k, n, p=pis)
    data = np.zeros(n)
    for i in range(n):
        k = z[i]
        data[i] = np.random.normal(mus[k], np.sqrt(variances[k]))

    return data


def main():
    """
    Apply EM-algorithm to finding mean, variance, and probability of latent variable values in
    mixture of Gaussians.
    Tutorials: 
    https://stephens999.github.io/fiveMinuteStats/intro_to_em.html
    http://bjlkeng.github.io/posts/the-expectation-maximization-algorithm/
    """
    num_k = 2
    mus, variances, pis = get_true_params(num_k)
    assert(num_k == len(mus))
    print("True params:", mus, variances, pis)

    n = 1000
    data = gen_data(n, mus, variances, pis)
    # print(data)
    plt.hist(data, bins=20)
    plt.show()

    mu_hats, var_hats, pi_hats = np.random.choice(data, 2, replace=False), np.array([1, 1]), np.ones(num_k)/num_k
    # mu_hats, var_hats, pi_hats = np.random.randint(1, 15, size=num_k), np.random.randint(1, 5, size=num_k), np.ones(num_k)/num_k
    print("Initial:", mu_hats, var_hats, pi_hats)
    eps = 0.001
    max_diff = np.inf

    mu_hat_list = []
    while max_diff > eps:
        gamma_iks = gamma_ik(mu_hats, var_hats, pi_hats, data)
        q_est = e_step(mu_hats, var_hats, pi_hats, data, gamma_iks)
        new_mu_hats, new_var_hats, new_pi_hats = m_step(mu_hats, data, q_est, gamma_iks)
        max_diff = np.amax([np.abs(new_mu_hats - mu_hats), np.abs(new_var_hats - var_hats), np.abs(new_pi_hats - pi_hats)])
        mu_hat_list.append(mu_hats)
        mu_hats = new_mu_hats
        var_hats = new_var_hats
        pi_hats = new_pi_hats

    mu_hat_list = np.array(mu_hat_list)
    for i in range(len(mu_hat_list[0])):
        plt.plot(range(len(mu_hat_list)),mu_hat_list[:,i] )
    plt.show()

    print(f"Final mu_hats: {mu_hats}")
    print(f"Final var_hats: {var_hats}")
    print(f"Final pi_hats: {pi_hats}")

def test_m_step():
    mu_hats = [1, 2]
    data = [1,2,3,4,5]
    q_est = 10
    gamma_iks = np.arange(10).reshape(5,2)
    # new_mu_hats
    assert(np.all(m_step(mu_hats, data, q_est, gamma_iks)[0] == np.array([4, 3.8])))
    
    # new_var_hats, given updated_mu_hats = [4, 3.8]
    assert(np.all(np.isclose(m_step(mu_hats, data, q_est, gamma_iks)[1], np.array([1.0, 1.8496]))))

    # new_pi_hats
    assert(np.all(m_step(mu_hats, data, q_est, gamma_iks)[2] == np.array([4, 5])))

def test_gamma_ik():
    data = [1,2,3,4,5]
    pi_hats = [0.25, 0.75]
    mu_hats = [0, 5]
    var_hats = [1, 2]
    gamma_iks = []
    for k in range(len(mu_hats)):
        gamma_i = ss.norm.pdf(data, loc=mu_hats[k], scale=np.sqrt(var_hats[k])) * pi_hats[k]
        gamma_iks.append(gamma_i)

    gamma_iks = np.array(gamma_iks)
    gamma_iks = (gamma_iks/np.sum(gamma_iks, axis=0)).T    

    assert(np.all(gamma_ik(mu_hats, var_hats, pi_hats, data) == gamma_iks))


def run_tests():
    test_gamma_ik()
    test_m_step()
    
if __name__ == "__main__":
    # run_tests()
    main()