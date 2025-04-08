import warnings

from ga import GeneticAlgorithm
from mcmc import MCMC
from models import *  # Import your null potential model class
import threading
from models.tests.test_model import TestModel

mcmc = MCMC(TestModel,ndim=1,lower_bound=-0.1,upper_bound=0.1,database=False)

mcmc.begin()