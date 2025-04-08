import warnings

from ga import GeneticAlgorithm
from mcmc import MCMC
from models import *  # Import your null potential model class
import threading

warnings.filterwarnings("ignore")
#


def main():
    # Create a GeneticAlgorithm instance using the null potential model.
    # The candidate parameter vector is assumed to be 6-dimensional (p_phi0, p_phi01, p_psi0, p_psi01, p_alpha, p_lam).
    ga = GeneticAlgorithm(
        model_class=M1V0,  # the model object to be used; no default is provided
        npar=6,
        varlo=-1,
        varhi=1,
        maxit=1000,
        popsize=100,
        threshold=0.1,
        mutrate=0.7,
        selection=0.5,
        stagnation_limit=10,
        print_filter_reason=False,
    )

    best_params, best_fitness = ga.run()

    print("Best parameters found:", best_params)
    print("Final fitness:", best_fitness)


if __name__ == "__main__":
    main()

mcmc_m1vphi2 = MCMC(
    model_class=M1VPHI2, nthreads=5, ndim=1, lower_bound=-1, upper_bound=1
)
mcmc_m1vphi24 = MCMC(
    model_class=M1VPHI24, nthreads=5, ndim=2, lower_bound=-3, upper_bound=3
)
mcmc_m1vexp = MCMC(
    model_class=M1VEXP, nthreads=5, ndim=2, lower_bound=-5, upper_bound=5
)

mcmc_m2vphi2 = MCMC(
    model_class=M2VPHI2, nthreads=5, ndim=1, lower_bound=-0.5, upper_bound=0.5
)
mcmc_m2vphi24 = MCMC(
    model_class=M2VPHI24, nthreads=5, ndim=2, lower_bound=-4, upper_bound=4
)
mcmc_m2vexp = MCMC(
    model_class=M2VEXP, nthreads=5, ndim=2, lower_bound=-5, upper_bound=5
)

mcmc_m3vphi2 = MCMC(
    model_class=M3VPHI2, nthreads=5, ndim=1, lower_bound=-0.5, upper_bound=0.5
)
mcmc_m3vphi24 = MCMC(
    model_class=M3VPHI24, nthreads=5, ndim=2, lower_bound=-4, upper_bound=4
)
mcmc_m3vexp = MCMC(
    model_class=M3VEXP, nthreads=5, ndim=2, lower_bound=-5, upper_bound=5
)

mcmc_threads = [
    threading.Thread(target=mcmc_m1vphi2.begin),
    threading.Thread(target=mcmc_m1vphi24.begin),
    threading.Thread(target=mcmc_m1vexp.begin),
]

mcmc_m2 = [
    threading.Thread(target=mcmc_m2vphi2.begin),
    threading.Thread(target=mcmc_m2vphi24.begin),
    threading.Thread(target=mcmc_m2vexp.begin),
]

mcmc_m3 = [
    threading.Thread(target=mcmc_m3vphi2.begin),
    threading.Thread(target=mcmc_m3vphi24.begin),
    threading.Thread(target=mcmc_m3vexp.begin),
]

for t in mcmc_threads:
    t.start()

for t in mcmc_m2:
    t.start()

for t in mcmc_m3:
    t.start()
