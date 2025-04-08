import sys

import numpy as np
import emcee
from schwimmbad import MultiPool
from db import DB
from models import Model
from models.potential import PotentialModel


class MCMC:
    def __init__(
        self,
        model_class=None,
        lower_bound=-5,
        upper_bound=5,
        ndim=None,
        nsteps=1000,
        nwalkers=128,
        nthreads=16,
        database=True,
    ):
        self.initials = model_class.get_initials()
        self.model_class: PotentialModel.__class__ = model_class
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.Yp_ave = 0.245
        self.Yp_std = 0.003

        self.DoH_ave = 2.547
        self.DoH_std = 0.029

        self.He3oH_ave = 1.08
        self.He3oH_std = 0.12

        self.Li7oH_ave = 1.6
        self.Li7oH_std = 0.3

        self.model: Model
        self.ndim = ndim
        self.nsteps = nsteps
        self.nwalkers = nwalkers
        self.nthreads = nthreads

        if database:
            self.runid = DB.get_next_runid(self.model_class.to_string())

        self.database = database

    def _get_time_str(self):
        import datetime

        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _log_likelihood(self, theta):
        Yp, DoH, He3oH, Li7oH = self.model.compute_abundances()

        chi2_Yp = (Yp - self.Yp_ave) ** 2 / self.Yp_std**2
        chi2_DoH = (DoH - self.DoH_ave) ** 2 / self.DoH_std**2
        chi2_He3oH = (He3oH - self.He3oH_ave) ** 2 / self.He3oH_std**2

        # save data
        # print(f"Valid parameters found for {self.model_class.to_string()}")
        # sys.stdout.flush()

        if self.database:
            DB.add_numbers(
                self.model_class.to_string(),
                self.runid,
                theta,
                goodness=-0.5 * (chi2_Yp + chi2_DoH + chi2_He3oH),
                date=self._get_time_str(),
            )

            DB.add_monte_carlo(
                self.model_class.to_string(),
                self.runid,
                [Yp, DoH, He3oH, Li7oH],
                logl=-0.5 * (chi2_Yp + chi2_DoH + chi2_He3oH),
                date=self._get_time_str(),
            )

        print(f"theta {theta} -> Yp: {Yp}, DoH: {DoH}, He3oH: {He3oH}, Li7oH: {Li7oH}")
        sys.stdout.flush()

        return -0.5 * (chi2_Yp + chi2_DoH + chi2_He3oH)

    def _log_prior(self, theta):
        constr = self.model.mcmc_constraints(theta)
        prior = np.all((theta >= self.lower_bound) & (theta <= self.upper_bound))
        if prior and constr and self.model.valid:
            return 0.0
        return -np.inf

    def _log_prob(self, theta):
        params = list(theta) + list(self.initials)
        self.model: Model = self.model_class(*params)

        lp = self._log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._log_likelihood(theta)

    def begin(self):
        with MultiPool(processes=self.nthreads) as pool:
            initial_pop = np.random.uniform(
                self.lower_bound, self.upper_bound, (self.nwalkers, self.ndim)
            )

            sampler = emcee.EnsembleSampler(
                self.nwalkers, self.ndim, self._log_prob, pool=pool
            )
            sampler.run_mcmc(initial_pop, self.nsteps, progress=True)
