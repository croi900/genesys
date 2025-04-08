import numpy as np

from models.potential import PotentialModel


class M3V0(PotentialModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, lambda_dpsi2=True)

    @staticmethod
    def get_initials():
        return [
            -0.00938917,
            -0.00093121,
            -0.08795092,
            0.0155224,
            0.02328965,
            -0.04948566,
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
        d_psi = psi_prime
        d_psi_2 = psi_prime2

        d_phi_2 = (
            -3 * h * phi_prime - self.beta * phi_prime**2 - self.beta * self._v(phi)
        )

        d_h = 0.5 * (
            -3 * h**2
            - rr
            - np.exp(self.beta * phi) * (0.5 * phi_prime**2 - self._v(phi))
            + (self.alpha**2 / 2) * (psi_prime2 + 3 * h * psi_prime)
        )
        h_prime = d_h

        # Left side = (β/6) φ′³ e^{βφ}
        # Right side = (α²/2) h ψ′² + (α²/2) d/dτ[ψ′′ + 3hψ′ + ψ′²]
        # We isolate: d/dτ[ψ′′ + 3hψ′ + ψ′²] = (β/3α²) φ′³ e^{βφ} - h ψ′²
        psi_source = (self.beta / (3 * self.alpha**2)) * phi_prime**3 * np.exp(
            self.beta * phi
        ) - h * psi_prime**2

        # d/dτ[ψ′′ + 3hψ′ + ψ′²] = ψ′′′ + 3h′ψ′ + 3hψ′′ + 2ψ′ψ′′
        d_psi_3 = psi_source - (
            3 * h_prime * psi_prime + 3 * h * psi_prime2 + 2 * psi_prime * psi_prime2
        )

        # dr_m/dτ = -4hr_m + (α²/3) d/dτ[ψ′′ + 3hψ′ + ψ′²]
        d_rr = -4 * h * rr + (self.alpha**2 / 3) * (
            d_psi_3
            + 3 * h_prime * psi_prime
            + 3 * h * psi_prime2
            + 2 * psi_prime * psi_prime2
        )

        return [d_h, phi_prime, d_phi_2, psi_prime, d_psi_2, d_psi_3, d_rr]

    @staticmethod
    def to_string():
        return "m3v0"
