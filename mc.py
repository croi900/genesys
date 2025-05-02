import random
import threading
import warnings

import dataset
import numpy as np

from db import DB
from ga import GeneticAlgorithm
from joblib import Parallel, delayed
from mcmc import MCMC
from models import *  # Import your null potential model class

warnings.filterwarnings("ignore")
class CallHolder:
    def __init__(self, clazz, runid, mu: float | tuple[float, float] = 0,
                 std: float | tuple[float, float] = 0, num_param=1, num_runs=1):
        self.model_class = clazz
        self.runid = runid
        self.mu = mu
        self.std = std
        self.num_param = num_param
        self.num_runs=num_runs

    def _combute_bbn_model(self, clazz, runid, mu, std, num_param, num_runs):
        def comp():
            np.random.seed(random.randint(1,10000000))
            if num_param == 1:
                # param = np.random.normal(mu, std, (1,))
                param = [mu]
            else:
                param = list(mu)#np.random.normal(mu, std, (2,))
            parameters = list(param) + list(clazz.get_initials())
            model = clazz(*parameters)
            if model.valid == False:
                print("Model is invalid, reasons: ", model.reasons)
                return
            res = model.compute_bbn()

            if res is not None:
                print(param)
                DB.add_bbn(
                    clazz.to_string(),
                    runid,
                    res, theta=str(param))
        Parallel(n_jobs=-1, verbose=55)(delayed(comp)() for _ in range(
            num_runs))


    def call(self):
        self._combute_bbn_model(self.model_class, self.runid, self.mu,
                                self.std, self.num_param, self.num_runs)

"""
m1vphi2: 0.0005615235277372262, 0.0001598178336046932
m2vphi2: 0.00024083959854368932, 8.766017124077335e-05
m3vphi2: 0.0003059104844155844, 0.00011275282324593558
m1vphi24: -0.005509549623843782, 0.0023895031821490665
m2vphi24: -0.000407619815153719, 0.00019527345867162578
m3vphi24: -3.0237504571428573e-05, 2.0157890494672115e-05
m1vphi24: 2.1985050873586847, 0.9584276393664122
m2vphi24: 0.3226697100165289, 0.12905472738339577
m3vphi24: 0.5838196933333333, 0.20160200591133598
m1vexp: 2.824530644374782, 0.05339503015951905
m2vexp: 2.8949689209401708, 0.051284796651672855
m3vexp: 2.970619408993576, 0.013936769624759645
m1vexp: 2.515918196865204, 1.1824084883834685
m2vexp: 1.5628342940170938, 0.6941707304567979
m3vexp: 1.693317008137045, 0.6312819082150594

"""
calls = [
    # CallHolder(M1VPHI2, 0,0.0005615235277372262, 0.0001598178336046932, 1, 10000),
    # CallHolder(M1VPHI24, 0, (-0.005509549623843782, 2.1985050873586847), (0.0023895031821490665, 0.9584276393664122), 2, 10000),
    # CallHolder(M1VEXP, 0, (2.824530644374782, 2.515918196865204), (0.05339503015951905, 1.1824084883834685), 2, 10000),
    #
    # CallHolder(M2VPHI2, 0,0.00024083959854368932, 8.766017124077335e-05, 1, 10000),
    # CallHolder(M2VPHI24, 0, (-0.000407619815153719, 0.3226697100165289), (0.00019527345867162578, 0.12905472738339577), 2, 10000),
    # CallHolder(M2VEXP, 0, (2.8949689209401708, 1.5628342940170938), (0.051284796651672855, 0.6941707304567979), 2, 10000),
    #
    CallHolder(M3VPHI2, 0,0.0003059104844155844, 0.00011275282324593558, 1, 10000),
    CallHolder(M3VPHI24, 0, (-3.0237504571428573e-05, 0.5838196933333333), (2.0157890494672115e-05, 0.20160200591133598), 2, 10000),
    CallHolder(M3VEXP, 0, (2.970619408993576, 1.693317008137045), (0.013936769624759645, 0.6312819082150594), 2, 10000)
]


threads = []
for call in calls:
    thread = threading.Thread(target=call.call)
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()