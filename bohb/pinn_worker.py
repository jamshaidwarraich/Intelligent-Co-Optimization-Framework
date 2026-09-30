import Pyro4
Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED = {"pickle"}
import hashlib
import random
import json
import os
import argparse
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import pearsonr

from hpbandster.core.worker import Worker
import ConfigSpace as CS

from pinn_model import PINNNet, pde_residual
from data_utils import load_embeddings_and_labels, setup_domain


class PhyFoldBOHBWorker(Worker):
    """
    BOHB Worker for PhyFold PINN
    - MATCHES PhyFold math: PDE + IC + BC + BIO
    - Uses BOHB budget as epochs (NO patience early stopping)
    - Returns VALIDATION RMSE as BOHB objective
    """

    def __init__(
        self,
        emb_path: str,
        lab_path: str,
        working_dir: str = "./bohb_temp",
        max_epochs: int = 50000,          # budget=1.0 => 50000 epochs
        seed: int = 1234,
        subset_size_eval: int = 200,      # test eval subset (info only)
        bio_batch_size: int = 64,         # fixed like your code
        N_f: int = 1500,
        N_bc: int =  200,
        lambda_pde: int = 10,
        lambda_ic: int = 100,
        lambda_bc: int =  50,
        lambda_data: int =  190,
        J: int = 50,
        dt: float = 0.1,
        Tmax: float = 1.0,
        # fixed env params
        T: float = 37.0,
        Topt: float = 37.0,
        pH: float = 7.4,
        pHopt: float = 7.4,
        P: float = 1.0,
        P0: float = 1.0,
        kappaT: float = 0.01,
        kappapH: float = 0.5,
        gammaP: float = 0.01,
        D: float = 0.01,
        alpha0: float = 2.0,
        beta0: float = 0.2,

        **kwargs
    ):
        super().__init__(**kwargs)

        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Worker {kwargs.get('id','unknown')} running on device: {self.device}")

        self.emb_path = emb_path
        self.lab_path = lab_path

        self.max_epochs = int(max_epochs)
        self.seed = int(seed)

        self.subset_size_eval = int(subset_size_eval)
        self.bio_batch_size = int(bio_batch_size)

        # default N_bc (BOHB config can override in compute/validation)
        self.N_bc = int(N_bc)

        self.J = int(J)
        self.dt = float(dt)
        self.Tmax = float(Tmax)

        # fixed env
        self.T = float(T)
        self.Topt = float(Topt)
        self.pH = float(pH)
        self.pHopt = float(pHopt)
        self.P = float(P)
        self.P0 = float(P0)
        self.kappaT = float(kappaT)
        self.kappapH = float(kappapH)
        self.gammaP = float(gammaP)
        self.D = float(D)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)

        # Load data once per worker
        self.data_dict = load_embeddings_and_labels(self.emb_path, self.lab_path, self.device)

        # Setup domain once per worker
        self._setup_domain_once()

        # Precompute env alpha/beta (fixed env)
        self.alpha, self.beta = self._compute_alpha_beta()

        # Torch constants
        self.D_t = torch.tensor(self.D, dtype=torch.float32, device=self.device)
        self.alpha_t = torch.tensor(self.alpha, dtype=torch.float32, device=self.device)
        self.beta_t = torch.tensor(self.beta, dtype=torch.float32, device=self.device)

    def _setup_domain_once(self):
        domain = setup_domain(self.J, self.dt, self.Tmax)

        self.a = domain["a"]
        self.b = domain["b"]
        self.x_grid = domain["x_grid"]  # (Nx,1) numpy
        self.t_grid = domain["t_grid"]  # (Nt,1) numpy
        self.Nx = domain["Nx"]
        self.Nt = domain["Nt"]
        self.u0_x = domain["u0_x"]

        # IC tensors
        x_ic_np = self.x_grid.copy()
        t_ic_np = np.zeros_like(x_ic_np)
        u_ic_np = self.u0_x.copy()

        self.x_ic = torch.tensor(x_ic_np, dtype=torch.float32, device=self.device)
        self.t_ic = torch.tensor(t_ic_np, dtype=torch.float32, device=self.device)
        self.u_ic = torch.tensor(u_ic_np, dtype=torch.float32, device=self.device)

    def _compute_alpha_beta(self):
        alpha = self.alpha0 * np.exp(-self.kappaT * (self.T - self.Topt) ** 2) * np.exp(-self.kappapH * (self.pH - self.pHopt) ** 2)
        beta = self.beta0 * (1 + self.gammaP * (self.P - self.P0))
        return float(alpha), float(beta)

    def _stable_config_seed(self, config):
        def to_py(v):
            import numpy as np
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            if isinstance(v, (np.str_,)):
                return str(v)
            return v

        cfg = {k: to_py(v) for k, v in dict(config).items()}
        s = json.dumps(cfg, sort_keys=True)
        h = hashlib.md5(s.encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    def _seed_all(self, extra_seed=0):
        s = self.seed + int(extra_seed)

        np.random.seed(s)
        random.seed(s)

        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    @torch.no_grad()
    def _bio_loss_batch(self, model, idxs):
        """
        BIO loss: (max_x u(x,Tmax) - label)^2 for a batch of embeddings.
        """
        X_all = self.data_dict["X_all"]
        y = self.data_dict["normalized_labels"]

        emb_batch = X_all[idxs]
        label_batch = torch.tensor(y[idxs], dtype=torch.float32, device=self.device)

        x_tensor = torch.tensor(self.x_grid, dtype=torch.float32, device=self.device)
        t_final = torch.full((self.Nx, 1), self.Tmax, dtype=torch.float32, device=self.device)

        B = emb_batch.shape[0]
        x_rep = x_tensor.repeat(B, 1)
        t_rep = t_final.repeat(B, 1)
        emb_rep = emb_batch.repeat_interleave(self.Nx, dim=0)

        u_pred = model(x_rep, t_rep, emb_rep).reshape(B, self.Nx, 1)
        readout = u_pred.max(dim=1).values.squeeze()
        return torch.mean((readout - label_batch) ** 2)

    def compute_validation_loss(self, model, config):
        """
        Validation loss:
        lambda_pde*PDE + lambda_ic*IC + lambda_bc*BC + lambda_data*BIO
        """
        model.eval()

        # deterministic validation sampling
        torch.manual_seed(0)
        np.random.seed(0)

        # ✅ FIXED
        lambda_pde = 10.0
        lambda_ic = 100.0
        lambda_bc = 50.0
        lambda_data = 190.0

        N_f = 1500
        N_bc = 200



        # PDE val points
        x_val = torch.rand(N_f, 1, device=self.device) * (self.b - self.a) + self.a
        t_val = torch.rand(N_f, 1, device=self.device) * self.Tmax

        val_idxs = self.data_dict["val_idxs"]
        replace_flag = len(val_idxs) < N_f
        idxs = np.random.choice(val_idxs, size=N_f, replace=replace_flag)
        emb_val = self.data_dict["X_all"][idxs]

        res_val = pde_residual(model, x_val, t_val, emb_val, self.D_t, self.alpha_t, self.beta_t)
        loss_pde_val = torch.mean(res_val ** 2)

        # IC val
        idxs_ic = np.random.choice(val_idxs, size=self.x_ic.shape[0], replace=True)
        emb_ic_val = self.data_dict["X_all"][idxs_ic]
        u_pred_ic = model(self.x_ic, self.t_ic, emb_ic_val)
        loss_ic_val = torch.mean((u_pred_ic - self.u_ic) ** 2)

        # BC val (Neumann) - IMPORTANT: use N_bc
        t_bc_np = np.random.rand(N_bc, 1) * self.Tmax
        t_bc = torch.tensor(t_bc_np, dtype=torch.float32, device=self.device, requires_grad=True)

        x_bc0 = torch.zeros_like(t_bc) + self.a
        x_bc0.requires_grad_(True)
        x_bc1 = torch.ones_like(t_bc) * self.b
        x_bc1.requires_grad_(True)

        idxs_bc = np.random.choice(val_idxs, size=N_bc, replace=True)
        emb_bc = self.data_dict["X_all"][idxs_bc]

        u0 = model(x_bc0, t_bc, emb_bc)
        ux0 = torch.autograd.grad(u0, x_bc0, torch.ones_like(u0), create_graph=False)[0]

        u1 = model(x_bc1, t_bc, emb_bc)
        ux1 = torch.autograd.grad(u1, x_bc1, torch.ones_like(u1), create_graph=False)[0]

        loss_bc_val = torch.mean(ux0 ** 2) + torch.mean(ux1 ** 2)

        # BIO val
        bs = min(self.bio_batch_size, len(val_idxs))
        idxs_bio = np.random.choice(val_idxs, size=bs, replace=False if len(val_idxs) >= bs else True)
        loss_bio_val = self._bio_loss_batch(model, idxs_bio)

        total = lambda_pde * loss_pde_val + lambda_ic * loss_ic_val + lambda_bc * loss_bc_val + lambda_data * loss_bio_val
        return float(total.item())


    def compute_validation_rmse(self, model):
        """
        Validation RMSE for BOHB objective.
        """
        model.eval()

        val_idxs = self.data_dict["val_idxs"]
        y_true = self.data_dict["normalized_labels"][val_idxs]
        y_pred = np.zeros(len(val_idxs))

        x_eval = torch.tensor(
            np.tile(self.x_grid.flatten(), (self.Nt, 1)).reshape(-1, 1),
            dtype=torch.float32, device=self.device
        )
        t_eval = torch.tensor(
            np.repeat(self.t_grid.flatten(), self.Nx).reshape(-1, 1),
            dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            for k, idx in enumerate(val_idxs):
                emb_seq = self.data_dict["X_all"][idx:idx+1]
                emb_rep = emb_seq.repeat(x_eval.shape[0], 1)
                u_pred = model(x_eval, t_eval, emb_rep).detach().cpu().numpy().reshape(self.Nt, self.Nx)
                y_pred[k] = u_pred[-1, :].max()

        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        return rmse

    def compute(self, config, budget, **kwargs):
        """
        BOHB objective:
        - Train for epochs = int(max_epochs * budget)
        - Return validation RMSE (lower is better)
        """
        try:
            cfg_seed = self._stable_config_seed(config) % 100000
            self._seed_all(extra_seed=cfg_seed)



            n_epochs = max(1, int(self.max_epochs * float(budget)))

            # hyperparams
            lr = float(config["learning_rate"])
            opt_name = str(config["optimizer"])
            # ✅ ADD THIS BLOCK HERE (VERY IMPORTANT)
            N_f = 1500
            N_bc = 200

            lambda_pde = 10.0
            lambda_ic = 100.0
            lambda_bc = 50.0
            lambda_data = 190.0

            # model architecture (optimized)
            emb_dim = int(self.data_dict["emb_dim"])
            hidden_dim = int(config["hidden_dim"])
            n_hidden = int(config["n_hidden_layers"])

            # layers: [input] + n_hidden*[hidden_dim] + [1]
            layers = [2 + emb_dim] + [hidden_dim] * n_hidden + [1]
            model = PINNNet(layers, emb_dim).to(self.device)

            # optimizer
            if opt_name == "Adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            elif opt_name == "RMSprop":
                optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
            elif opt_name == "AdamW":
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            else:
                raise ValueError(f"Unknown optimizer: {opt_name}")

            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1000)

            train_idxs = self.data_dict["train_idxs"]

            last_val = None
            for epoch in range(n_epochs):
                # collocation points
                x_f = torch.rand(N_f, 1, device=self.device) * (self.b - self.a) + self.a
                x_f.requires_grad_(True)
                t_f = torch.rand(N_f, 1, device=self.device) * self.Tmax
                t_f.requires_grad_(True)

                # boundary points (Neumann) - IMPORTANT: use N_bc
                t_bc_np = np.random.rand(N_bc, 1) * self.Tmax
                t_bc = torch.tensor(t_bc_np, dtype=torch.float32, device=self.device, requires_grad=True)

                x_bc0 = torch.zeros_like(t_bc) + self.a
                x_bc0.requires_grad_(True)
                x_bc1 = torch.ones_like(t_bc) * self.b
                x_bc1.requires_grad_(True)

                # embeddings sampling
                idx_f = np.random.choice(train_idxs, size=N_f, replace=True)
                emb_f = self.data_dict["X_all"][idx_f]

                idx_ic = np.random.choice(train_idxs, size=self.x_ic.shape[0], replace=True)
                emb_ic = self.data_dict["X_all"][idx_ic]

                idx_bc = np.random.choice(train_idxs, size=N_bc, replace=True)
                emb_bc = self.data_dict["X_all"][idx_bc]

                optimizer.zero_grad()
                model.train()

                # PDE loss
                res_f = pde_residual(model, x_f, t_f, emb_f, self.D_t, self.alpha_t, self.beta_t)
                loss_pde = torch.mean(res_f ** 2)

                # IC loss
                u_pred_ic = model(self.x_ic, self.t_ic, emb_ic)
                loss_ic = torch.mean((u_pred_ic - self.u_ic) ** 2)

                # BC loss (Neumann)
                u0 = model(x_bc0, t_bc, emb_bc)
                ux0 = torch.autograd.grad(u0, x_bc0, torch.ones_like(u0), create_graph=True)[0]
                u1 = model(x_bc1, t_bc, emb_bc)
                ux1 = torch.autograd.grad(u1, x_bc1, torch.ones_like(u1), create_graph=True)[0]
                loss_bc = torch.mean(ux0 ** 2) + torch.mean(ux1 ** 2)

                # BIO loss
                bs = min(self.bio_batch_size, len(train_idxs))
                idx_bio = np.random.choice(train_idxs, size=bs, replace=False if len(train_idxs) >= bs else True)
                loss_bio = self._bio_loss_batch(model, idx_bio)

                # total loss
                loss = lambda_pde * loss_pde + lambda_ic * loss_ic + lambda_bc * loss_bc + lambda_data * loss_bio

                if not torch.isfinite(loss):
                    return {"loss": 1e10, "info": {"error": "loss became NaN/Inf", "config": dict(config), "budget": float(budget)}}

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # validate periodically
                if epoch % 500 == 0 or epoch == (n_epochs - 1):
                    last_val = self.compute_validation_rmse(model)
                    scheduler.step(last_val)

            final_rmse = float(last_val) if last_val is not None else self.compute_validation_rmse(model)

            # Optional: quick test metrics (info only)
            test_metrics = {}
            try:
                test_idxs = self.data_dict["test_idxs"]
                subset = min(self.subset_size_eval, len(test_idxs))
                eval_idxs = np.random.choice(test_idxs, size=subset, replace=False)

                y_true = self.data_dict["normalized_labels"][eval_idxs]
                y_pred = np.zeros(len(eval_idxs))

                x_eval = torch.tensor(
                    np.tile(self.x_grid.flatten(), (self.Nt, 1)).reshape(-1, 1),
                    dtype=torch.float32, device=self.device
                )
                t_eval = torch.tensor(
                    np.repeat(self.t_grid.flatten(), self.Nx).reshape(-1, 1),
                    dtype=torch.float32, device=self.device
                )

                model.eval()
                with torch.no_grad():
                    for k, idx in enumerate(eval_idxs):
                        emb_seq = self.data_dict["X_all"][idx:idx+1]
                        emb_rep = emb_seq.repeat(x_eval.shape[0], 1)
                        # reshape to (Nt, Nx) so "last time slice" is [-1, :]
                        u_pred_flat = model(x_eval, t_eval, emb_rep).detach().cpu().numpy().reshape(self.Nt, self.Nx)
                        y_pred[k] = u_pred_flat[-1, :].max()

                y_t = np.array(y_true)
                y_p = np.array(y_pred)

                rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
                mae = float(np.mean(np.abs(y_t - y_p)))
                ss_tot = float(np.sum((y_t - y_t.mean()) ** 2))
                r2 = float(1 - (np.sum((y_t - y_p) ** 2) / ss_tot)) if ss_tot > 0 else float("nan")
                pr = float(pearsonr(y_t, y_p)[0])

                test_metrics = {"rmse": rmse, "mae": mae, "r2": r2, "pearson_r": pr}
            except Exception:
                test_metrics = {"note": "test metrics failed (non-critical)"}

            print("\n==============================")
            print("CONFIG FINISHED")
            print("Budget:", budget)
            print("Epochs:", n_epochs)
            print("Validation RMSE:", final_rmse)
            print("Config:", config)
            print("==============================\n")

            return {
                "loss": final_rmse,
                "info": {
                    "budget": float(budget),
                    "epochs": int(n_epochs),
                    "alpha": float(self.alpha),
                    "beta": float(self.beta),
                    "val_rmse": final_rmse,
                    "test_metrics": test_metrics,
                    "config": dict(config),
                },
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"loss": 1e10, "info": {"error": str(e), "budget": float(budget)}}


def create_configspace():
    cs = CS.ConfigurationSpace()

    # Architecture (extra optimization)
    cs.add_hyperparameter(CS.CategoricalHyperparameter("hidden_dim", choices=[32, 64, 128, 256], default_value=64))
    cs.add_hyperparameter(CS.CategoricalHyperparameter("n_hidden_layers", choices=[2, 3, 4], default_value=2))

    # Optimizer/LR
    cs.add_hyperparameter(CS.UniformFloatHyperparameter("learning_rate", lower=1e-5, upper=1e-3, default_value=5e-4, log=True))
    cs.add_hyperparameter(CS.CategoricalHyperparameter("optimizer", choices=["Adam", "RMSprop","AdamW"], default_value="Adam"))
    return cs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default="phyfold_optimization")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--ns_host", type=str, required=True)
    parser.add_argument("--ns_port", type=int, required=True)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--emb_path", type=str, required=True)
    parser.add_argument("--lab_path", type=str, required=True)
    parser.add_argument("--working_dir", type=str, default="./bohb_temp")
    parser.add_argument("--max_epochs", type=int, default=50000)
    args = parser.parse_args()

    w = PhyFoldBOHBWorker(
        run_id=args.run_id,
        host=args.host,
        nameserver=args.ns_host,
        nameserver_port=args.ns_port,
        id=f"phyfold_worker_{args.worker_id}",
        emb_path=args.emb_path,
        lab_path=args.lab_path,
        working_dir=args.working_dir,
        max_epochs=args.max_epochs,
    )
    w.run()