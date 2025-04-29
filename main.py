import os
import threading
import warnings
from multiprocessing import freeze_support

import dataset
from schwimmbad import MultiPool

from ga import GeneticAlgorithm
from mcmc import MCMC
from models import *  # Import your null potential model class

from pool import DaskPool
from dask.distributed import LocalCluster, Client
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    cluster = None


    mcmc_m1vphi2 = MCMC(
        model_class=M1VPHI2, nthreads=5, ndim=1, lower_bound=-1, upper_bound=1, cluster=cluster
    )
    mcmc_m1vphi24 = MCMC(
        model_class=M1VPHI24, nthreads=5, ndim=2, lower_bound=-3, upper_bound=3, cluster=cluster
    )
    mcmc_m1vexp = MCMC(
        model_class=M1VEXP, nthreads=5, ndim=2, lower_bound=-5, upper_bound=5, cluster=cluster
    )

    mcmc_m2vphi2 = MCMC(
        model_class=M2VPHI2, nthreads=5, ndim=1, lower_bound=-0.5, upper_bound=0.5, cluster=cluster
    )
    mcmc_m2vphi24 = MCMC(
        model_class=M2VPHI24, nthreads=5, ndim=2, lower_bound=-4, upper_bound=4, cluster=cluster
    )
    mcmc_m2vexp = MCMC(
        model_class=M2VEXP, nthreads=5, ndim=2, lower_bound=-5, upper_bound=5, cluster=cluster
    )

    mcmc_m3vphi2 = MCMC(
        model_class=M3VPHI2, nthreads=5, ndim=1, lower_bound=-0.5, upper_bound=0.5, cluster=cluster
    )
    mcmc_m3vphi24 = MCMC(
        model_class=M3VPHI24, nthreads=5, ndim=2, lower_bound=-4, upper_bound=4, cluster=cluster
    )
    mcmc_m3vexp = MCMC(
        model_class=M3VEXP, nthreads=5, ndim=2, lower_bound=-5, upper_bound=5, cluster=cluster
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
