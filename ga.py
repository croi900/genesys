import numpy as np
import random
import traceback
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
from joblib import Parallel, delayed
import os

from models import *


class GeneticAlgorithm:
    def __init__(
        self,
        model_class,
        npar,
        varlo,
        varhi,
        maxit,
        popsize,
        threshold,
        mutrate,
        selection,
        stagnation_limit,
        freezeout_lower=3.068,
        freezeout_threshold=0.00135,
        print_filter_reason=False,
    ):
        self.model_class = model_class
        self.npar = npar
        self.varlo = varlo
        self.varhi = varhi
        self.maxit = maxit
        self.popsize = popsize
        self.threshold = threshold
        self.mutrate = mutrate
        self.selection = selection
        self.stagnation_limit = stagnation_limit
        self.freezeout_lower = freezeout_lower
        self.freezeout_threshold = freezeout_threshold
        self.print_filter_reason = print_filter_reason
        self.reasons_path = (
            f"logs/reasons_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        )

        warnings.filterwarnings("ignore")

    def plot_results(self, model_obj, generation):
        t = model_obj.theta
        h = model_obj.h
        phi = model_obj.phi
        psi = model_obj.psi
        rr = model_obj.rr

        try:
            ww = model_obj.time.p_w(t) / model_obj.time.rho_w(t)
        except Exception:
            ww = np.zeros_like(t)

        plt.figure(figsize=(12, 8))
        plt.subplot(3, 2, 1)
        plt.plot(t, h, label="h(t)")
        plt.xlabel("Time")
        plt.legend()

        plt.subplot(3, 2, 2)
        plt.plot(t, phi, label="φ(t)")
        plt.xlabel("Time")
        plt.legend()

        plt.subplot(3, 2, 3)
        plt.plot(t, psi, label="ψ(t)")
        plt.xlabel("Time")
        plt.legend()

        plt.subplot(3, 2, 4)
        plt.plot(t, rr, label="rr(t)")
        plt.xlabel("Time")
        plt.legend()

        plt.subplot(3, 2, 5)
        try:
            rho_vals = model_obj.time.rho_w(t)
        except Exception:
            rho_vals = np.zeros_like(t)
        plt.plot(t, rho_vals, label="ρ_w(t)")
        plt.xlabel("Time")
        plt.legend()

        plt.subplot(3, 2, 6)
        plt.plot(t, ww, label="w/w(t)")
        plt.xlabel("Time")
        plt.legend()

        plt.tight_layout()

        if not os.path.exists("figs"):
            os.makedirs("figs")

        plt.savefig(f"figs/ga_results_gen{generation}.pdf")
        plt.close()

    def works_with_potential(self, candidate, model_class):
        try:
            prefix = model_class.to_string()[:2].lower()
            if prefix == "m1":
                cls_phi2, cls_phi24, cls_exp = M1VPHI2, M1VPHI24, M1VEXP
            elif prefix == "m2":
                cls_phi2, cls_phi24, cls_exp = M2VPHI2, M2VPHI24, M2VEXP
            elif prefix == "m3":
                cls_phi2, cls_phi24, cls_exp = M3VPHI2, M3VPHI24, M3VEXP
            else:
                return None  # Unknown model prefix

            thetas = list(np.random.uniform(low=-3, high=3, size=1)) + list(candidate)
            model_phi2 = cls_phi2(*thetas)
            if not model_phi2.valid:
                return None

            thetas = list(np.random.uniform(low=-3, high=3, size=2)) + list(candidate)
            model_phi24 = cls_phi24(*thetas)
            if not model_phi24.valid:
                return None

            thetas = list(np.random.uniform(low=-3, high=3, size=2)) + list(candidate)
            model_exp = cls_exp(*thetas)
            if not model_exp.valid:
                return None

            return np.mean(
                [
                    model_exp.time.rho_w(3.068),
                    model_phi2.time.rho_w(3.066),
                    model_phi24.time.rho_w(3.066),
                ]
            )
        except Exception:
            return None

    def try_works_with_potential(candidate, model_class):
        print(f"Evaluating works_with_potential for {candidate}")
        results = Parallel(n_jobs=4)(
            delayed(works_with_potential)(candidate, model_class) for _ in range(100)
        )
        return True in results

    def validate_candidate(self, candidate):
        try:
            model_instance = self.model_class(*candidate)
            if not model_instance.valid:
                reasons = (
                    model_instance.reasons
                    if hasattr(model_instance, "reasons")
                    else ["Model invalid"]
                )
                return False, reasons, model_instance

            if not np.all(np.diff(model_instance.phi) < 0):
                return False, ["phi is not strictly decreasing"], model_instance

            if not np.all(model_instance.rr > -0.001):
                return False, ["rr values are too negative"], model_instance
            if not try_works_with_potential(candidate, self.model_class):
                return False, ["failed in try_works_with_potential"], model_instance

            return True, [], model_instance
        except Exception as e:
            return False, [str(e)], None

    def param_fitness(self, candidate):
        valid, reasons, model_instance = self.validate_candidate(candidate)
        if not valid:
            return None
        try:
            t = model_instance.theta

            ww_vals = model_instance.time.p_w(t) / model_instance.time.rho_w(t)
            ww_penalty = np.sum(np.abs(ww_vals[~((ww_vals > -1) & (ww_vals < 1))]))

            freezeout_val = model_instance.temp.rho_w(self.freezeout_lower)
            freezeout_penalty = max(0, freezeout_val - self.freezeout_threshold)

            rr_penalty = np.abs(np.sum(model_instance.rr[model_instance.rr < 0]))

            phi_penalty = np.abs(np.sum(model_instance.phi[model_instance.phi < 0]))

            avg_rho = np.mean(model_instance.rho_w)
            rho_mag_penalty = 1 / avg_rho if avg_rho != 0 else 1e6

            total_penalty = (
                ww_penalty
                + freezeout_penalty
                + rr_penalty
                + phi_penalty
                + rho_mag_penalty
            )
            return total_penalty
        except Exception:
            return None

    def filter_valid_population(self, population):
        results = Parallel(n_jobs=8)(
            delayed(self.validate_candidate)(candidate) for candidate in population
        )
        valid_population = []
        for candidate, (is_valid, reasons, _) in zip(population, results):
            if is_valid:
                valid_population.append(candidate)
            else:
                if self.print_filter_reason:
                    print(
                        f"Candidate {candidate} filtered out due to: {', '.join(reasons)}"
                    )
                else:
                    if not os.path.exists("logs"):
                        os.makedirs("logs")

                    reason_file = open(self.reasons_path, "a")

                    reason_file.write(
                        f"Candidate {candidate} filtered out due to: {', '.join(reasons)}\n"
                    )
        if not valid_population:
            print("No valid candidates found! Generating a new random population.")
            return np.random.uniform(
                self.varlo, self.varhi, size=(self.popsize, self.npar)
            )
        return np.array(valid_population)

    def generate_valid_population(self):
        while True:
            print("Generating a new population...")
            pop = np.random.uniform(
                self.varlo, self.varhi, size=(self.popsize, self.npar)
            )
            pop = self.filter_valid_population(pop)
            fits = Parallel(n_jobs=-1)(
                delayed(self.param_fitness)(candidate) for candidate in pop
            )
            valid_pop_and_fits = [(p, f) for p, f in zip(pop, fits) if f is not None]
            if len(valid_pop_and_fits) >= 2:
                valid_pop_and_fits.sort(key=lambda x: x[1])
                pop_sorted = np.array([x[0] for x in valid_pop_and_fits])
                fits_sorted = np.array([x[1] for x in valid_pop_and_fits])
                return pop_sorted, fits_sorted
            else:
                print("Not enough valid candidates. Regenerating population...\n")

    def rank_select_indices(self, keep_size):
        probs = np.flipud(
            np.arange(1, keep_size + 1) / np.sum(np.arange(1, keep_size + 1))
        )
        odds = np.cumsum(probs)

        def select():
            r = random.random()
            return next(i for i, val in enumerate(odds) if r <= val)

        return select

    def single_point_crossover(self, parent1, parent2):
        point = random.randint(1, self.npar - 1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    def mutate(self, population_slice):
        for i in range(len(population_slice)):
            if random.random() < self.mutrate:
                col = random.randint(0, self.npar - 1)
                population_slice[i, col] = np.random.uniform(self.varlo, self.varhi)
        return population_slice

    def run(self):
        pop, fits = self.generate_valid_population()
        best_fitness = fits[0]
        best_candidate = pop[0].copy()
        stagnation_counter = 0
        generation = 0

        while generation < self.maxit:
            try:
                current_size = len(pop)
                if current_size < 2:
                    print("Population too small for crossover. Regenerating.")
                    pop, fits = self.generate_valid_population()
                    continue

                keep = max(2, int(self.selection * current_size))
                parent_pop = pop[:keep]
                select_fn = self.rank_select_indices(keep)
                M = int(keep * 0.33)
                if M < 1:
                    print(
                        "Not enough parents to produce offspring. Regenerating population."
                    )
                    pop, fits = self.generate_valid_population()
                    continue

                offspring = []
                for i in range(M):
                    idx1 = select_fn()
                    idx2 = select_fn()
                    p1 = parent_pop[idx1]
                    p2 = parent_pop[idx2]
                    child1, child2 = self.single_point_crossover(p1, p2)
                    offspring.append(child1)
                    offspring.append(child2)
                offspring = np.array(offspring)
                offspring = self.mutate(offspring)
                new_offspring = self.filter_valid_population(offspring)
                new_fits = Parallel(n_jobs=-1)(
                    delayed(self.param_fitness)(candidate)
                    for candidate in new_offspring
                )
                valid_offspring = [
                    (child, f)
                    for child, f in zip(new_offspring, new_fits)
                    if f is not None
                ]

                combined = list(zip(pop, fits)) + valid_offspring
                combined.sort(key=lambda x: x[1])
                pop = np.array([x[0] for x in combined])[: self.popsize]
                fits = np.array([x[1] for x in combined])[: self.popsize]

                current_best = fits[0]
                if current_best < best_fitness:
                    best_fitness = current_best
                    best_candidate = pop[0].copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                generation += 1
                print(
                    f"Generation {generation}: Best fitness = {current_best}, Mean fitness = {np.mean(fits)}"
                )

                if best_fitness < self.threshold:
                    print(f"Convergence achieved with fitness {best_fitness}")
                    break

                if stagnation_counter >= self.stagnation_limit:
                    print(
                        f"No improvement for {self.stagnation_limit} generations. Stopping."
                    )
                    break
            except Exception:
                traceback.print_exc()
                continue

        valid, _, model_instance = self.validate_candidate(best_candidate)
        if valid and model_instance is not None:
            self.plot_results(model_instance, generation)

        return best_candidate, best_fitness
