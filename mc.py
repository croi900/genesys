import threading
import warnings

import dataset
from db import DB
from ga import GeneticAlgorithm
from joblib import Parallel, delayed
from mcmc import MCMC
from models import *  # Import your null potential model class

warnings.filterwarnings("ignore")


class CallHolder:
    def __init__(self, clazz, runid):
        self.model_class = clazz
        self.runid = runid

    def _combute_bbn_model(self, clazz, runid):
        db = dataset.connect(
            "mysql+pymysql://weyl:strongpassword@192.168.1.137"
            "/weyl")
        table = db[clazz.to_string()]
        print(clazz.to_string())
        rows = table.find(runid=runid)

        if "phi24" not in clazz.to_string() and 'exp' not in clazz.to_string():
            params = [[row['x']] for row in rows]
        else:
            params = [[row['x'], row['z']] for row in rows]

        def comp(param):

            parameters = list(param) + list(clazz.get_initials())
            model: PotentialModel = clazz(*parameters)
            if model.valid == False:
                return
            res = model.compute_bbn()

            if res is not None:
                print(param)
                DB.add_bbn(
                    clazz.to_string(),
                    runid,
                    res)

        Parallel(n_jobs=2)(delayed(comp)(param) for param in params)

    def call(self):
        self._combute_bbn_model(self.model_class, self.runid)


calls = [CallHolder(M1VPHI24, 0),
         CallHolder(M1VEXP, 0),
         CallHolder(M2VPHI2, 0),
         CallHolder(M2VPHI24, 0),
         CallHolder(M2VEXP, 0),
         CallHolder(M3VPHI2, 0),
         CallHolder(M3VPHI24, 0),
         CallHolder(M3VEXP, 0)]

threads = []
for call in calls:
    thread = threading.Thread(target=call.call)
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()