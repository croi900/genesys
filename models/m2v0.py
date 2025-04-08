import numpy as np

from models.potential import PotentialModel


class M2V0(PotentialModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, lambda_dpsi2=True)

    @staticmethod
    def get_initials():
        return [
            -0.03686985,
            -0.01737777,
            0.04948915,
            -0.04746112,
            0.0238886,
            0.03729411,
        ]

    def _compute_beta(self):
        self.beta = (
            np.log(
                3
                - self.alpha**2
                / 6
                * (self.lam + 3 * self.h0 * self.psi01 + self.psi01**2)
            )
            - np.log(self.phi01**2 / 2 + 0)
        ) / self.phi0

    def _system(self, t, Y):
        h, phi, phi_prime, psi, psi_prime, psi_prime2, rr = Y

        d_phi = phi_prime

        d_phi_2 = (
            -3 * h * phi_prime
            - self._v(phi)
            - self.beta * phi_prime**2
            - self.beta * self._v(phi)
        )

        d_psi = psi_prime
        d_psi_2 = psi_prime2

        d_h = 0.5 * (
            -3 * h**2
            - rr
            - np.exp(self.beta * phi) * (0.5 * phi_prime**2 - self._v(phi))
            + (self.alpha**2 / 2) * (psi_prime2 + 3 * h * psi_prime)
        )

        h_prime = d_h

        term1 = 3 * h_prime * psi_prime
        term2 = 3 * h * d_psi
        term3 = 2 * psi_prime * d_psi
        term4 = 3 * h * psi_prime**2

        d_psi_3 = -(term1 + term2 + term3 + term4)

        d_rr = -4 * h * rr + (self.beta / 6) * phi_prime**3 * np.exp(self.beta * phi)

        return [d_h, d_phi, d_phi_2, d_psi, d_psi_2, d_psi_3, d_rr]

    @staticmethod
    def to_string():
        return "m2v0"
