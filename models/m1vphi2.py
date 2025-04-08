import numpy as np

from models import M1V0
from models.potential import PotentialModel


class M1VPHI2(M1V0):
    def __init__(self, x, *args, **kwargs):
        self.x = x
        super().__init__(*args, **kwargs)

    def _v(self, phi):
        return self.x * phi**2

    def _compute_beta(self):
        self.beta = (
            np.log(3 - self.alpha**2 / 2 * self.lam)
            - np.log(self.phi01**2 / 2 + self.x * self.phi0**2)
        ) / self.phi0

    def _system(self, t, Y):
        h, phi, phi_prime, psi, psi_prime, rr = Y

        d_phi_2 = (
            -3 * h * phi_prime
            - 2 * self.x * phi
            - self.beta * (phi_prime**2)
            - self.beta * self._v(phi)
        )

        d_phi = phi_prime
        d_psi = psi_prime

        d_psi_2 = -3 * h * psi_prime - psi_prime**2 + self.lam

        d_h = 0.5 * (
            -3 * h**2
            - rr
            - np.exp(self.beta * phi) * (0.5 * phi_prime**2 - self._v(phi))
            + (self.alpha**2 / 2) * (self.lam - psi_prime**2)
        )

        d_rr = (
            -4 * h * rr
            + (self.beta / 6) * phi_prime**3 * np.exp(self.beta * phi)
            - (self.alpha**2 / 2) * h * psi_prime**2
        )

        return [d_h, phi_prime, d_phi_2, psi_prime, d_psi_2, d_rr]

    @staticmethod
    def mcmc_constraints(theta):
        # mass must always be positive
        # v(phi) = m/2 * phi ^ 2
        if theta[0] < 0:
            return False
        else:
            return True

    @staticmethod
    def to_string():
        return "m1vphi2"
