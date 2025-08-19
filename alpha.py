import numpy as np
import random
import traceback
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
from joblib import Parallel, delayed
import subprocess
import dill
import pickle
import tempfile
import os
import time
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from models import *
#Model 1 Epoch Best: 1610.323876904468 | All-Time Best: 1981.8652124977523 -> [0.01383253 0.21218537 1.         1.         0.05145449 0.23121775]
#Model 1.2 Epoch Best: 1580.9813852736668 | All-Time Best: 1873.5461697096657 -> [0.02434534 0.58118246 1.         1.         0.01932736 0.1303598 ]
class AlphaGA:
    def __init__(
        self,
        model_class,
        npar,
        varlo,
        varhi,
        maxit,
        popsize,
        mc_runs,
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
        self.mc_runs = mc_runs
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




                

    
    def validate_candidate(self, candidate):
        #print("validating")
        
        # Generate unique filenames for this validation
        unique_id = str(uuid.uuid4())
        input_file = os.path.join(tempfile.gettempdir(), f"validation_input_{unique_id}.pkl")
        output_file = os.path.join(tempfile.gettempdir(), f"validation_output_{unique_id}.pkl")
        
        # Create a temporary script to run the validation
        script_content = f'''
import dill  # Use dill instead of pickle
import numpy as np
import sys
import os
import traceback

# Add the current working directory to path to import your modules
sys.path.insert(0, "{os.getcwd()}")

try:
    # Load the serialized data
    with open("{input_file}", "rb") as f:
        model_class, candidate = dill.load(f)
    
    #print("Loaded data successfully", file=sys.stderr)
    
    model_instance = model_class(*candidate)
    print("Created model instance", file=sys.stderr)
    
    if not model_instance.valid:
        reasons = (
            model_instance.reasons
            if hasattr(model_instance, "reasons")
            else ["Model invalid"]
        )
        result = (False, reasons, model_instance)
    elif not np.all(np.diff(model_instance.phi) < 0):
        result = (False, ["phi is not strictly decreasing"], model_instance)
    elif not np.all(model_instance.rr > -0.001):
        result = (False, ["rr values are too negative"], model_instance)
    else:
        result = (True, [], model_instance)
    
    #print("Validation complete, saving result", file=sys.stderr)
    
    # Save result
    with open("{output_file}", "wb") as f:
        dill.dump(result, f)
        
    #print("Result saved successfully", file=sys.stderr)
        
except Exception as e:
    #print(f"Exception occurred: {{str(e)}}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    try:
        with open("{output_file}", "wb") as f:
            dill.dump((False, [str(e)], None), f)
    except Exception as e2:
        print(f"Failed to save error result: {{str(e2)}}", file=sys.stderr)
    '''
        
        try:
            import dill
            
            # Serialize input data with dill
            with open(input_file, "wb") as f:
                dill.dump((self.model_class, candidate), f)
            
            # Remove output file if it exists
            if os.path.exists(output_file):
                os.remove(output_file)
            
            # Run validation in subprocess with timeout
            process = subprocess.Popen([
                sys.executable, "-c", script_content
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            try:
                stdout, stderr = process.communicate(timeout=10)
                
                # Debug output
                if stderr:
                    #print(f"Subprocess stderr: {stderr.decode()}")
                    pass
                
                # Load result
                if os.path.exists(output_file):
                    with open(output_file, "rb") as f:
                        result = dill.load(f)
                    return result
                else:
                    return False, [f"Validation failed - no output. Process exit code: {process.returncode}"], None
                    
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                return False, ["Validation timed out"], None
                
        except ImportError:
            return False, ["dill package required for robust validation"], None
        except Exception as e:
            return False, [f"Validation setup failed: {str(e)}"], None
        finally:
            # Cleanup
            for file in [input_file, output_file]:
                try:
                    if os.path.exists(file):
                        os.remove(file)
                except:
                    pass

    def test_sigma(self, sigmas):
        total_good = 0
        means = self.model_class.get_initials()
        
        def test():
            params = [np.random.normal(mu,std) for mu,std in zip(means,sigmas)]
            valid, messages,_ = self.validate_candidate(params)
            #print(messages)
            return 1 if valid else 0
                
        total_good = np.sum(Parallel(n_jobs = 15)(delayed(test)() for _ in range(self.mc_runs)))

        return np.sum([(sigma/self.varlo) * (total_good/self.mc_runs) for sigma in sigmas]), sigmas

    def generate_valid_population(self):
        return np.random.uniform(self.varlo, self.varhi, size=(self.popsize, self.npar))

    def mutate(self, candidate):
        for i in range(len(candidate)):
            if random.random() < self.mutrate:
                col = random.randint(0, self.npar - 1)
                candidate[i] = np.random.uniform(self.varlo, self.varhi) / 2 + candidate[i] * (1 + 2*(np.random.uniform() - 0.5)/10) / 2
                candidate[i] = min(self.varhi,max(candidate[i], self.varlo))
        return candidate

    def corssover(self, parent1, parent2):
        point = random.randint(1, self.npar - 1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1

    def eval_population(self, popultaion):

        results =  Parallel(n_jobs=2)(delayed(self.test_sigma)(candidate) for candidate in popultaion)
        results = list(sorted(results, key=lambda x: x[0], reverse=True))
        # print(results)

        return results

    def run(self):
        pop = self.generate_valid_population()
        bestest = [0,None]
        for _ in range(self.stagnation_limit):
           # Inside the run method's loop
           

            sorted_results = self.eval_population(pop).copy()
            best = sorted_results[:(int(len(sorted_results) * self.selection))].copy()
            print(len(best))
            #reporting

            if bestest[0] < best[0][0]:
                bestest = best[0]
                print(f"NEW ALL-TIME BEST: {bestest[0]}") # Report when a new best is found

            print(pop[0])

            print(f"Epoch Best: {best[0][0]} | All-Time Best: {bestest[0]} -> {bestest[1]}")


            if bestest[0] < best[0][0]:
                bestest = best[0]

            #breeding
            offspring = []
            elite_pool = [b[1] for b in best]
            while len(offspring) + len(elite_pool) < self.popsize:
                parent1, parent2 = random.sample(elite_pool, 2)
                child = self.corssover(parent1, parent2)
                offspring.append(child)
            pop = elite_pool + offspring

            #mutate 

            for i in range(1, len(pop)):
                pop[i] = self.mutate(pop[i])




      



if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    ga = AlphaGA(M3V0, 6,0.001, 1, 1000, 100, 100, 0.3, 0.4, 1000)
    ga.run()



