###########run_bohb_optimization.py
import os
import time
import json
import pickle
import random
import argparse
import numpy as np
import pandas as pd
import torch
import multiprocessing as mp

import Pyro4
Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED = {"pickle"}
Pyro4.config.COMPRESSION = True

import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB

from pinn_worker import PhyFoldBOHBWorker, create_configspace


# ===============================
# ARGUMENT PARSER
# ===============================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BOHB hyperparameter search for the PhyFold PINN."
    )

    parser.add_argument("--emb_path", type=str, default="data/kpro_embeddings.npy",
                         help="Path to the protein embeddings (.npy file).")
    parser.add_argument("--lab_path", type=str, default="data/kpro_labels.csv",
                         help="Path to the labels (.csv file).")
    parser.add_argument("--work_dir", type=str, default="bohb_results",
                         help="Directory where BOHB results will be saved.")

    parser.add_argument("--run_id", type=str, default="phyfold_bohb",
                         help="Identifier for this BOHB run.")

    parser.add_argument("--eta", type=int, default=3,
                         help="Successive halving parameter (eta).")
    parser.add_argument("--min_budget", type=float, default=0.1,
                         help="Minimum budget (fraction of max epochs).")
    parser.add_argument("--max_budget", type=float, default=1.0,
                         help="Maximum budget (fraction of max epochs).")

    parser.add_argument("--n_workers", type=int, default=2,
                         help="Number of parallel BOHB workers.")
    parser.add_argument("--n_iterations", type=int, default=60,
                         help="Number of BOHB iterations (HyperBand brackets).")

    parser.add_argument("--seed", type=int, default=1234,
                         help="Base random seed for reproducibility.")

    return parser.parse_args()


# ===============================
# SEED FUNCTION (REPRODUCIBILITY)
# ===============================
def seed_everything(seed=1234):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


# ===============================
# JSON SAFE CONVERTER
# ===============================
def to_jsonable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.str_):
        return str(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    return obj


# ===============================
# WORKER LAUNCH FUNCTION
# ===============================
def run_worker(worker_id, run_id, ns_host, ns_port, emb_path, lab_path, work_dir):

    worker = PhyFoldBOHBWorker(
        run_id=run_id,
        host="127.0.0.1",
        nameserver=ns_host,
        nameserver_port=ns_port,
        id=f"phyfold_worker_{worker_id}",
        emb_path=emb_path,
        lab_path=lab_path,
        working_dir=os.path.join(work_dir, f"worker_{worker_id}"),
    )

    worker.run()


# ===============================
# MAIN FUNCTION
# ===============================
def main():

    args = parse_args()

    seed_everything(args.seed)

    print("=" * 70)
    print(" PhyFold BOHB Optimization")
    print("=" * 70)

    EMB_PATH = args.emb_path
    LAB_PATH = args.lab_path
    WORK_DIR = args.work_dir

    os.makedirs(WORK_DIR, exist_ok=True)

    RUN_ID = args.run_id

    eta = args.eta
    min_budget = args.min_budget
    max_budget = args.max_budget

    n_workers = args.n_workers
    n_iterations = args.n_iterations

    configspace = create_configspace()

    print("\n Configuration:")
    print(f"run_id      = {RUN_ID}")
    print(f"seed        = {args.seed}")
    print(f"budgets     = [{min_budget}, {max_budget}]")
    print(f"eta         = {eta}")
    print(f"workers     = {n_workers}")
    print(f"iterations  = {n_iterations}")
    print(f"embeddings  = {EMB_PATH}")
    print(f"labels      = {LAB_PATH}")
    print(f"results_dir = {WORK_DIR}")

    if not os.path.exists(EMB_PATH):
        raise FileNotFoundError(
            f"Embeddings file not found at: {EMB_PATH}\n"
            f"Pass the correct path with --emb_path."
        )
    if not os.path.exists(LAB_PATH):
        raise FileNotFoundError(
            f"Labels file not found at: {LAB_PATH}\n"
            f"Pass the correct path with --lab_path."
        )

    # ===============================
    # START NAMESERVER
    # ===============================
    print("\n Starting NameServer...")

    NS = hpns.NameServer(run_id=RUN_ID, host="127.0.0.1", port=0)
    ns_host, ns_port = NS.start()

    print(f"✅ NameServer running on {ns_host}:{ns_port}")

    # ===============================
    # START WORKERS
    # ===============================
    print(f"\n Launching {n_workers} workers...")

    processes = []

    for i in range(n_workers):

        p = mp.Process(
            target=run_worker,
            args=(i, RUN_ID, ns_host, ns_port, EMB_PATH, LAB_PATH, WORK_DIR)
        )

        p.start()

        processes.append(p)

        print(f" Worker {i} launched (PID: {p.pid})")

        time.sleep(1)

    # ===============================
    # CREATE BOHB MASTER
    # ===============================
    print("\n Creating BOHB Master...")

    bohb = BOHB(
        configspace=configspace,
        run_id=RUN_ID,
        host="127.0.0.1",
        nameserver=ns_host,
        nameserver_port=ns_port,
        min_budget=min_budget,
        max_budget=max_budget,
        eta=eta,
    )

    print("\n" + "=" * 70)
    print(" STARTING BOHB OPTIMIZATION")
    print("=" * 70 + "\n")

    start_time = time.time()

    try:

        result = bohb.run(
            n_iterations=n_iterations,
            min_n_workers=n_workers
        )

        # ===============================
        # SAVE RESULTS
        # ===============================

        ts = time.strftime("%Y%m%d-%H%M%S")

        SAVE_DIR = os.path.join(
            WORK_DIR,
            f"bohb_saved_{RUN_ID}_{ts}"
        )

        os.makedirs(SAVE_DIR, exist_ok=True)

        # Save full result object
        pkl_path = os.path.join(SAVE_DIR, "bohb_result.pkl")

        with open(pkl_path, "wb") as f:
            pickle.dump(result, f)

        # ===============================
        # SAVE ALL RUNS CSV
        # ===============================

        runs = result.get_all_runs()
        id2conf = result.get_id2config_mapping()

        rows = []

        for r in runs:

            cfg = id2conf[r.config_id]["config"]

            row = {
                "config_id": r.config_id,
                "budget": float(r.budget),
                "loss": float(r.loss) if r.loss is not None else None,
                "time_start": r.time_stamps.get("started", None),
                "time_end": r.time_stamps.get("finished", None),
            }

            for k, v in cfg.items():
                row[f"hp_{k}"] = v

            rows.append(row)

        df = pd.DataFrame(rows)

        csv_path = os.path.join(SAVE_DIR, "bohb_runs.csv")

        df.to_csv(csv_path, index=False)

        # ===============================
        # SAVE INCUMBENT TRAJECTORY
        # ===============================

        traj = result.get_incumbent_trajectory()

        traj_path = os.path.join(
            SAVE_DIR,
            "bohb_incumbent_trajectory.json"
        )

        with open(traj_path, "w") as f:
            json.dump(to_jsonable(traj), f, indent=2)

        print("\n BOHB results saved to:", SAVE_DIR)
        print("   -", pkl_path)
        print("   -", csv_path)
        print("   -", traj_path)

        # ===============================
        # BEST CONFIG
        # ===============================

        traj = result.get_incumbent_trajectory()

        best_config_id = traj["config_ids"][-1]

        best_loss = float(traj["losses"][-1])

        best_config = result.get_id2config_mapping()[best_config_id]["config"]

        print("\n BEST CONFIG FOUND")
        print("Best loss:", best_loss)
        print("Best config:", best_config)

        out_path = os.path.join(WORK_DIR, "bohb_best.json")

        with open(out_path, "w") as f:
            json.dump(
                {
                    "best_loss": best_loss,
                    "best_config": to_jsonable(best_config)
                },
                f,
                indent=2
            )

        print(" Best config saved:", out_path)

        elapsed = time.time() - start_time

        print(f"\n✅ Optimization completed in {elapsed:.1f} seconds")

    finally:

        print("\n Cleaning up...")

        try:
            bohb.shutdown(shutdown_workers=True)
        except:
            pass

        try:
            NS.shutdown()
        except:
            pass

        for p in processes:
            p.terminate()

        print("Cleanup complete")


# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    main()