import traceback
import warnings

from ..model import Model
import PRyM.PRyM_init as PRyMini
import PRyM.PRyM_main as PRyMmain
import numpy as np


class TestModel(Model):
    def __init__(self, a):
        self.a = a
        super().__init__()

    def rho(self, Tg):
        return self.a * Tg**4

    def p(self, Tg):
        return (1 / 3) * self.a * Tg**4

    def drho_dt(self, Tg):
        return self.a * 4 * Tg**3

    def compute_abundances(self):
        print("Computing abundances")
        PRyMini.NP_e_flag = True
        PRyMini.numba_flag = True
        PRyMini.nacreii_flag = True
        PRyMini.aTid_flag = False
        PRyMini.smallnet_flag = True
        PRyMini.compute_nTOp_flag = False
        PRyMini.recompute_nTOp_rates = False
        PRyMini.ReloadKeyRates()
        warnings.filterwarnings("error")
        try:
            prym = PRyMmain.PRyMclass(self.rho, self.p, self.drho_dt)
            warnings.filterwarnings("ignore")

            return prym.PRyMresults()[4:8]
        except:
            # traceback.print_exc()
            warnings.filterwarnings("ignore")

            return [1e7, 1e7, 1e7, 1e7]
