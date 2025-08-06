import numpy as np

from models.potential import PotentialModel


class M1V0(PotentialModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_initials():
        return [
            -0.0260986,
            -0.37956477,
            -0.2597126,
            0.0737921,
            -0.18993857,
            0.56484734,
        ]

    def _system(self, t, Y):
        h, phi, phi_prime, psi, psi_prime, rr = Y

        # d²φ/dτ² + 3h dφ/dτ + v'(φ) = -β (dφ/dτ)^2 - β v(φ)
        d_phi_2 = (
            -3 * h * phi_prime
            - self._v(phi)
            - self.beta * (phi_prime**2)
            - self.beta * self._v(phi)
        )

        # d²ψ/dτ² + 3h dψ/dτ + (dψ/dτ)^2 = λ
        d_psi_2 = -3 * h * psi_prime - psi_prime**2 + self.lam

        # 2 dh/dτ + 3h^2 = -r_m - exp(βφ)[½ (dφ/dτ)^2 - v(φ)] + (α^2 / 2)[λ - (dψ/dτ)^2]
        d_h = 0.5 * (
            -3 * h**2
            - rr
            - np.exp(self.beta * phi) * (0.5 * phi_prime**2 - self._v(phi))
            + (self.alpha**2 / 2) * (self.lam - psi_prime**2)
        )

        # dr_m/dτ + 4h r_m = (β / 6)(dφ/dτ)^3 e^{βφ} - (α^2 / 2) h (dψ/dτ)^2
        d_rr = (
            -4 * h * rr
            + (self.beta / 6) * phi_prime**3 * np.exp(self.beta * phi)
            - (self.alpha**2 / 2) * h * psi_prime**2
        )

        return [d_h, phi_prime, d_phi_2, psi_prime, d_psi_2, d_rr]

    @staticmethod
    def to_string():
        return "m1v0"
