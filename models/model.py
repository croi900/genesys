import traceback
import warnings

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.interpolate import interp1d
import PRyM.PRyM_init as PRyMini
import PRyM.PRyM_main as PRyMmain
from PRyM.PRyM_init import MeV, MeV4_to_gcmm3

class Model:
    def __init__(self):
        self.valid = True
        return

    def compute_abundances(self):
       return

    @staticmethod
    def mcmc_constraints(theta):
        return True

    @staticmethod
    def to_string():
        return "bald model"


    @staticmethod
    def get_initials():
        return []