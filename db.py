import socket
import io
import sys
import traceback

import numpy as np
import matplotlib.pyplot as plt
import requests
import dataset
from dotenv import dotenv_values
from termcolor import colored


class DB:
    db_url = "mysql+pymysql://weyl:strongpassword@192.168.1.137/weyl_oop"
    config = dotenv_values(".env")
    print(config)
    db = dataset.connect(db_url)

    @staticmethod
    def add_statistical(
        model: str, runid, stats, initial_runid=None, initial_param_id=None, date=None
    ):
        stats = list(stats)
        data = {
            "model": model,
            "runid": runid,
            "date": date,
            "initial_runid": initial_runid,
            "initial_param_id": initial_param_id,
        }

        tuples = [(k, float(v)) for k, v in zip(DB.config["STATS"].split(","), stats)]
        data.update(dict(tuples))
        with DB.db as tx:
            tx["stats"].insert(data)

    @staticmethod
    def add_numbers(
        model: str, runid, data_dict, initial_runid=None, goodness=None, date=None
    ):
        data_dict = list(data_dict)
        data = {
            "model": model,
            "runid": runid,
            "date": date,
            "initial_runid": initial_runid,
            "goodness": goodness,
        }
        tuples = [(f"theta{i}", float(v)) for i, v in enumerate(data_dict)]
        data.update(dict(tuples))
        with DB.db as tx:
            tx[model].insert(data)

    @staticmethod
    def add_monte_carlo(model: str, runid, abundances, date=None, logl=None):
        data = {
            "model": f"{model}",
            "runid": runid,
            "date": date,
            "Yp": abundances[0],
            "DoH": abundances[1],
            "He3oH": abundances[2],
            "Li7oH": abundances[3],
            "logl": logl,
        }
        with DB.db as tx:
            tx[f"{model}_mc"].insert(data)

    @staticmethod
    def update_row_goodness(model: str, id, goodness):
        with DB.db as tx:
            tx[model].update({"id": id, "goodness": float(goodness)}, ["id"])

    @staticmethod
    def get(model: str, runid):
        with DB.db as tx:
            rows = list(tx[model].find(runid=runid))
            if not rows:
                return []
            columns = list(rows[0].keys())
            result = [[row[col] for col in columns] for row in rows]
            return result

    @staticmethod
    def get_next_runid(model: str):
        try:
            with DB.db as tx:
                max_runid = -1
                rows = list(tx[model].find())
                if not rows:
                    return 0
                for row in rows:
                    if int(row["runid"]) > max_runid:
                        max_runid = int(row["runid"])
                return max_runid + 1
        except:
            print(colored("ERROR GETTING NEXT RUNID", "red"))
            traceback.print_exc()
            return 0
