import traceback
import warnings

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.interpolate import interp1d
import PRyM.PRyM_init as PRyMini
import PRyM.PRyM_main as PRyMmain
from PRyM.PRyM_init import MeV, MeV4_to_gcmm3
from .model import Model
# an object to help with
# notation of functions in the model
# framework


# will allow usage such as:
# model = M1V0(...)
# model.time.rho_w
class FuncPack:
    def __init__(self):
        self.rho_w = None
        self.p_w = None
        self.drho_dt_w = None


class PotentialModel(Model):
    def __init__(
        self, p_phi0, p_phi01, p_psi0, p_psi01, p_alpha, p_lam, lambda_dpsi2=False
    ):
        super().__init__()
        self.rtol = 1e-3  # ivp precision (relative)
        self.atol = 1e-6  # ivp precision (absolute)
        self.h0 = 1.0  # initial constant value for adimensional h
        self.rr0 = 0.0  # initial constant value for adimensional rr
        self.num_points = 20000  # number of points for IVP integration
        self.solver = "BDF"  # solver method for IVP
        self.theta = np.linspace(0, 10, self.num_points)

        # member values for initital parameters given
        self.phi0 = p_phi0
        self.phi01 = p_phi01
        self.psi0 = p_psi0
        self.psi01 = p_psi01
        self.alpha = p_alpha
        self.lam = p_lam

        self._compute_beta()

        # creating an initial guess ndarray
        # depending on model

        self.initial_guess = [
            self.h0,  # h
            self.phi0,  # φ
            self.phi01,  # dφ/dτ
            self.psi0,  # ψ
            self.psi01,  # dψ/dτ
            self.rr0,  # rr
        ]

        if lambda_dpsi2:
            self.initial_guess[-1] = self.lam
            self.initial_guess.append(self.rr0)

        self.initial_guess = np.array(self.initial_guess)

        # presetting IVP results to none for safety
        # in case somehow we return a valid object on failure
        # the code fails afterwards anyway
        self.h = None
        self.phi = None
        self.dphi = None
        self.psi = None
        self.dpsi = None
        self.rr = None

        # presetting validity to false and an obvious reason given program
        # flow :D
        self.valid = False
        self.reasons = ["Could not solve the IVP, error in sovle_ivp. "]

        # intializing funcpack objects
        # for stroing time dependent functions
        # and temperature dependent functions
        self.temp = FuncPack()
        self.time = FuncPack()

        # now we are solving!
        # on [0,10]s
        solve_result, solve_message = self._solve()
        if not solve_result:
            self.reasons[0] += solve_message
            return

        # first after the ivp is solved,
        # we compute time dependent functions
        # on the interval [0,10]s
        self._compute_functions_time()

        # we then extend the functions to
        # 1e7 (max PRyM time) and then
        # compute the temperature-time dependance
        self._compute_time_map()

        # we finally create callable
        # function by interpolating the
        # extended([0,1e7]s) time functions. for
        # temperature dependance, the approach is:
        # f_temp(T) = f_time(tofT(t))

        self._interpolate()

        # compute the freezout value for GA
        self._compute_freezout()

        # before creating the object, we update
        # it's validity status based on the
        # tests implemented below, and optionally
        # if the tests failed or something else went wrong
        # a reasons list is returned.
        self.valid, self.reasons = self._results_valid()

    def _compute_beta(self):
        self.beta = (
            np.log(3.0 - self.alpha**2 / 2.0 * self.lam) - np.log(self.phi01**2 / 2.0)
        ) / self.phi0

    # potential, by default null, can be overriden
    def _v(self, phi):
        return 0.0

    # system, by default empty, as it is an abstract class
    # requires implementation in each model
    def _system(self, t, Y):
        return

    # the solve ivp wrapper that helps us streamline functionality
    def _solve(self):
        try:
            warnings.filterwarnings("ignore")

            t_span = (self.theta[0], self.theta[-1])
            solution = solve_ivp(
                lambda t, Y: self._system(t, Y),
                t_span,
                self.initial_guess,
                t_eval=self.theta,
                method=self.solver,
                rtol=self.rtol,
                atol=self.atol,
            )

            if (
                solution == None
                or np.isnan(solution.y).any()
                or np.isinf(solution.y).any()
            ):
                return False, "Solution was none"

            if solution.success == False:
                return False, solution.message

            self.h = solution.y[0]
            self.phi = solution.y[1]
            self.dphi = solution.y[2]
            self.psi = solution.y[3]
            self.dpsi = solution.y[4]
            self.rr = solution.y[-1]

            return True, ""
        except:
            import sys, os

            log_file = os.path.join("logs", self.__class__.__name__ + "_logs")
            with open(log_file, "a") as f:
                traceback.print_exc(file=f)
                f.write(str(sys.exc_info()[1]) + "\n")

            return False, "Exception encountered while solving ivp."

    # computing time dependent functions
    def _compute_functions_time(self):
        rho_phi = np.exp(self.beta * self.phi) * (
            0.5 * self.dphi**2 + self._v(self.phi)
        )
        p_phi = np.exp(self.beta * self.phi) * (0.5 * self.dphi**2 - self._v(self.phi))

        dpsi2_f = np.gradient(self.dpsi)

        kinetic_psi = 0.5 * (dpsi2_f + 3 * self.h * self.dpsi + self.dpsi**2)

        self.rho_w = rho_phi + (self.alpha**2 / 2) * kinetic_psi
        self.p_w = p_phi - (self.alpha**2 / 2) * (dpsi2_f + 3 * self.h * self.dpsi)
        self.drho_dt_w = np.gradient(self.rho_w)

    # computing temperature-time dependance
    def _compute_time_map(self):
        self.theta_extended = np.concatenate(
            (self.theta, np.linspace(self.theta[-1] + 100, 1e7, 100000))
        )

        self.h_extended = np.concatenate(
            (
                self.h,
                np.ones_like(
                    self.theta_extended[len(self.h) : len(self.theta_extended)]
                )
                * self.h[-1],
            )
        )

        self.rho_w_extended = np.concatenate(
            (
                self.rho_w,
                np.ones_like(
                    self.theta_extended[len(self.rho_w) : len(self.theta_extended)]
                )
                * self.rho_w[-1],
            )
        )

        self.p_w_extended = np.concatenate(
            (
                self.p_w,
                np.ones_like(
                    self.theta_extended[len(self.p_w) : len(self.theta_extended)]
                )
                * self.p_w[-1],
            )
        )

        self.drho_dt_w_extended = np.concatenate(
            (
                self.drho_dt_w,
                np.ones_like(
                    self.theta_extended[len(self.p_w) : len(self.theta_extended)]
                )
                * self.drho_dt_w[-1],
            )
        )

        h_int_f = cumulative_trapezoid(
            self.h_extended, self.theta_extended, initial=0.007386520369079301
        )
        T0 = 10
        T_vals = T0 * np.exp(-h_int_f)

        self.tofT = interp1d(
            T_vals[::-1],
            self.theta_extended[::-1],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        self.Toft = interp1d(
            self.theta_extended,
            T_vals,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

    # interpolation
    def _interpolate(self):
        self.time.rho_w = self._lerp(self.theta_extended, self.rho_w_extended)
        self.time.p_w = self._lerp(self.theta_extended, self.p_w_extended)
        self.time.drho_dt_w = self._lerp(self.theta_extended, self.drho_dt_w_extended)

        self.temp.rho_w = (
            lambda t: self.time.rho_w(self.tofT(t)) * MeV**4 * MeV4_to_gcmm3
        )
        self.temp.p_w = lambda t: self.time.p_w(self.tofT(t)) * MeV**4 * MeV4_to_gcmm3
        self.temp.drho_dt_w = (
            lambda t: self.time.drho_dt_w(self.tofT(t)) * MeV**4 * MeV4_to_gcmm3
        )

    # wrapper around np.interp for interpolation, could as well use
    # interp1d with linear kind but this was faster
    def _lerp(self, ts, ys):
        def interpolation(t):
            return np.interp(t, ts, ys)

        return interpolation

    # ========= TESTS DOWN HERE =============

    # tests if p_w/rho_w is between [-1,1]
    # the actual test is done with a margin
    # of error of 0.08 to account for ivp precision
    def _ww_ratio_test(self):
        valid_times = self.theta[self.theta > 3.068]
        ww = self.time.p_w(valid_times) / self.time.rho_w(valid_times)
        return np.all(np.abs(ww) < 1.08)

    # checks if rr is positive, again
    # accounting for precision errors
    def _rr_test(self):
        return np.all(self.rr > -0.001)

    # computes the freezouts value
    def _compute_freezout(self):
        return self.temp.rho_w(0.5)

    # wrapper for all tests
    # togheter with null checks
    def _results_valid(self):
        reasons = []

        if not self._rr_test():
            reasons.append("RR test failed")
        if not self._ww_ratio_test():
            reasons.append("WW ratio test failed")
        if self.temp.rho_w is None:
            reasons.append("Rho_w is None")
        if self.temp.p_w is None:
            reasons.append("P_w is None")
        if self.temp.drho_dt_w is None:
            reasons.append("Drho_dt_w is None")
        if self.time.rho_w is None:
            reasons.append("Rho_w is None")
        if self.time.p_w is None:
            reasons.append("P_w is None")
        if self.time.drho_dt_w is None:
            reasons.append("Drho_dt_w is None")

        if len(reasons) == 0:
            return True, None
        else:
            return False, reasons

    def compute_abundances(self):
        if (
            self.temp.rho_w is None
            or self.temp.p_w is None
            or self.temp.drho_dt_w is None
        ):
            raise RuntimeError("Cannot compute abundances without potential")

        PRyMini.NP_e_flag = True
        PRyMini.numba_flag = True
        PRyMini.nacreii_flag = True
        PRyMini.aTid_flag = False
        PRyMini.smallnet_flag = True
        PRyMini.compute_nTOp_flag = False
        PRyMini.recompute_nTOp_rates = False
        PRyMini.ReloadKeyRates()
        try:
            prym = PRyMmain.PRyMclass(
                self.temp.rho_w, self.temp.p_w, self.temp.drho_dt_w
            )
            return prym.PRyMresults()[4:8]
        except:
            return [np.inf, np.inf, np.inf, np.inf]

    @staticmethod
    def mcmc_constraints(theta):
        return True
