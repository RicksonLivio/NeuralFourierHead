"""
Experimento principal da v3 (E1-v3): acurácia preditiva e estrutura das previsões nos
8 conjuntos públicos (4 fontes x {U, M}), 5 sementes (cinco treinamentos independentes).

Para cada conjunto, backbone b in {MLP, LSTM, Conv1D+LSTM} e semente s:
  1. treina b + camada densa do zero (MSE; Adam; parada antecipada na validação);
  2. para cada K in {10, 4, 2} (q = 2):
     a. decomposição pura: projeta a previsão densa em T, S, R (Teorema de identificabilidade)
        e mede a cobertura kappa; a previsão não muda;
     b. projeção: descarta o resíduo, y_proj = y_densa - R (kappa = 1 por construção, sem treino);
     c. NFH: conversão exata da camada densa (Teorema de equivalência; passo 0 = decomposição pura)
        + ajuste estrutural com a penalização do resíduo; 'NFH' = seleção segura (RMSE de validação
        nunca maior que o da densa); 'NFH (estrutura completa)' = menor objetivo de validação sem a
        restrição, para comparar com a projeção na mesma cobertura.
Mais o N-BEATS-I (decomposição nativa e projetada) e três previsões ingênuas.

Protocolo fixado após um piloto em ETTh1-U (results/v3_piloto/), que mostrou: (i) com K = 10 a
previsão densa já tem cobertura de 98-99,8%; (ii) escolher a época de maior cobertura levava
sistematicamente à última época. Por isso a escolha passou a ser pelo objetivo de validação e
acrescentou-se a ablação de K.

Uso: python src/v3_benchmark.py CONJ [CONJ ...]   (um processo por grupo de conjuntos)
Saídas (retomáveis): results/v3/runs/<conj>.csv, results/v3/partes/*.npz,
results/v3/modelos/*__s0.pt, results/v3/previsoes/*__s0.npy
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from v2_common import InstanceNorm, NBeatsI, set_seed, stack_ds, train_model  # noqa: E402
from v3_common import (BACKBONES, CFG3, H3, L3, Q3, RES3, SEEDS, Projetor, ajuste_estrutural,  # noqa: E402
                       avaliar, carregar, construir, densa_para_nfh, ingenuas, partes_nativas, segmentos)

K_LISTA = (10, 4, 2)
DIR_R, DIR_P, DIR_M, DIR_Y = RES3 / "runs", RES3 / "partes", RES3 / "modelos", RES3 / "previsoes"
for d_ in (DIR_R, DIR_P, DIR_M, DIR_Y):
    d_.mkdir(parents=True, exist_ok=True)
COLS = ["conjunto", "fonte", "modo", "modelo", "variante", "backbone", "K", "seed", "RMSE", "MAE", "val_RMSE",
        "epocas", "tempo_s", "params", "cobertura_mediana", "eps_T_mediana", "eps_S_mediana", "eps_R_mediana",
        "escala_mediana", "epoca_ajuste", "kappa_val", "kappa_val_densa", "rmse_val_densa", "erro_conversao",
        "n_treino", "n_val", "n_teste", "n_teste_total", "d"]
PROJ = {K: Projetor(H3, K, Q3) for K in K_LISTA}


def arquivo(nome):
    return DIR_R / f"{nome}.csv"


def feitos(nome):
    f = arquivo(nome)
    if not f.exists():
        return set()
    with f.open() as fh:
        return {(r["modelo"], r["variante"], r["K"], int(r["seed"])) for r in csv.DictReader(fh)}


def gravar(nome, linha):
    f = arquivo(nome)
    novo = not f.exists()
    with f.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if novo:
            w.writeheader()
        w.writerow({c: linha.get(c, "") for c in COLS})


def metricas(P, Y):
    e = P - Y
    return float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e)))


def resumo(P, K):
    pr = PROJ[K].partes(P)
    return pr, {"cobertura_mediana": float(np.median(1 - pr["eps_R"])),
                "eps_T_mediana": float(np.median(pr["eps_T"])), "eps_S_mediana": float(np.median(pr["eps_S"])),
                "eps_R_mediana": float(np.median(pr["eps_R"])), "escala_mediana": float(np.median(pr["escala"]))}


def n_params(m):
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad))


@torch.no_grad()
def prever(modelo, X, dev):
    modelo.eval()
    return torch.cat([modelo(X[i:i + 1024].to(dev))["forecast"].cpu() for i in range(0, len(X), 1024)]).double().numpy()


def rodar(nome, dev):
    import os
    # Filtros opcionais para dividir um conjunto entre processos sem sobreposição:
    # V3_BACKBONES="MLP,LSTM" restringe os backbones; V3_NBEATS="0" pula o N-BEATS-I.
    bbs = [b for b in os.environ.get("V3_BACKBONES", ",".join(BACKBONES)).split(",") if b]
    com_nbeats = os.environ.get("V3_NBEATS", "1") == "1"
    # V3_SEEDS="2,3,4" restringe as sementes (backbones e N-BEATS-I); sem a variável, usa todas.
    sem = os.environ.get("V3_SEEDS", "")
    seeds = [int(x) for x in sem.split(",") if x.strip()] if sem.strip() else list(SEEDS)
    ja = feitos(nome)
    ds = carregar(nome)
    tr, va, te = segmentos(ds)
    Xv, Yv = stack_ds(va)
    Xte, Yte = stack_ds(te)
    Y = Yte.double().numpy()
    np.save(DIR_Y / f"{nome}__real.npy", Y.astype(np.float32))
    base = {"conjunto": nome, "fonte": ds["fonte"], "modo": ds["modo"], "n_treino": len(tr), "n_val": len(va),
            "n_teste": len(te), "n_teste_total": te.n_total, "d": ds["feats"].shape[1]}
    print(f"[{time.strftime('%H:%M:%S')}] {nome}: d={base['d']} janelas {len(tr)}/{len(va)}/{len(te)}", flush=True)
    for mod, P in ingenuas(Xte, H3).items():
        if (mod, "", "", -1) not in ja:
            r, a = metricas(P, Y)
            gravar(nome, {**base, "modelo": mod, "variante": "", "K": "", "seed": -1, "RMSE": r, "MAE": a})
    for bb in bbs:
        for s in seeds:
            if (f"{bb} + NFH", "seguro", str(K_LISTA[-1]), s) in ja:
                continue
            set_seed(s)
            m = construir(bb, "densa", base["d"])
            t0 = time.time()
            m, info = train_model(m, tr, va, s, lam=0.0, cfg=CFG3, dev=dev)
            t_d = time.time() - t0
            rv = avaliar(m, Xv, Yv, dev)[0]
            Pd = prever(m, Xte, dev)
            rd, ad = metricas(Pd, Y)
            linha_d = {**base, "backbone": bb, "seed": s, "params": n_params(m)}
            gravar(nome, {**linha_d, "modelo": f"{bb} + densa", "variante": "", "K": "", "RMSE": rd, "MAE": ad,
                          "val_RMSE": rv, "epocas": info["epochs"], "tempo_s": t_d, **resumo(Pd, 10)[1]})
            if s == 0:
                torch.save(m.state_dict(), DIR_M / f"{nome}__{bb}+densa__s0.pt")
                np.save(DIR_Y / f"{nome}__{bb}+densa__s0.npy", Pd.astype(np.float32))
            partes = {}
            for K in K_LISTA:
                prd, sd = resumo(Pd, K)
                _, _, R = PROJ[K].decompor(Pd)
                Pp = Pd - R
                rp, ap = metricas(Pp, Y)
                gravar(nome, {**linha_d, "modelo": f"{bb} + densa", "variante": "decomposição pura", "K": K,
                              "RMSE": rd, "MAE": ad, "val_RMSE": rv, **sd})
                gravar(nome, {**linha_d, "modelo": f"{bb} + densa", "variante": "projeção", "K": K, "RMSE": rp,
                              "MAE": ap, **resumo(Pp, K)[1]})
                nfh = densa_para_nfh(m, K=K, q=Q3, H=H3)
                with torch.no_grad():
                    b = Xv[:512].to(dev)
                    err = float((nfh(b)["forecast"] - m(b)["forecast"]).abs().max())
                nfh, est_completo, aj = ajuste_estrutural(nfh, tr, va, s, dev=dev)
                Pn = prever(nfh, Xte, dev)
                rn, an = metricas(Pn, Y)
                prn, sn = resumo(Pn, K)
                gravar(nome, {**linha_d, "modelo": f"{bb} + NFH", "variante": "seguro", "K": K, "RMSE": rn, "MAE": an,
                              "val_RMSE": aj["seguro"]["rmse"], "epocas": aj["seguro"]["epoca"],
                              "tempo_s": aj["tempo_ajuste_s"], "params": n_params(nfh), **sn,
                              "epoca_ajuste": aj["seguro"]["epoca"], "kappa_val": aj["seguro"]["kappa"],
                              "kappa_val_densa": aj["kappa_val_densa"], "rmse_val_densa": aj["rmse_val_densa"],
                              "erro_conversao": err})
                if s == 0 and K == 10:
                    torch.save(nfh.state_dict(), DIR_M / f"{nome}__{bb}+NFH__s0.pt")
                    np.save(DIR_Y / f"{nome}__{bb}+NFH__s0.npy", Pn.astype(np.float32))
                nfh.load_state_dict(est_completo)
                Pc = prever(nfh, Xte, dev)
                rc, ac = metricas(Pc, Y)
                gravar(nome, {**linha_d, "modelo": f"{bb} + NFH", "variante": "completo", "K": K, "RMSE": rc,
                              "MAE": ac, "val_RMSE": aj["completo"]["rmse"], "epocas": aj["completo"]["epoca"],
                              "params": n_params(nfh), **resumo(Pc, K)[1], "epoca_ajuste": aj["completo"]["epoca"],
                              "kappa_val": aj["completo"]["kappa"], "kappa_val_densa": aj["kappa_val_densa"],
                              "rmse_val_densa": aj["rmse_val_densa"]})
                partes.update({f"K{K}_densa_eps_T": prd["eps_T"], f"K{K}_densa_eps_R": prd["eps_R"],
                               f"K{K}_nfh_eps_T": prn["eps_T"], f"K{K}_nfh_eps_S": prn["eps_S"],
                               f"K{K}_nfh_eps_R": prn["eps_R"],
                               f"K{K}_trajetoria": np.array(aj["trajetoria"], dtype=float)})
                print(f"[{time.strftime('%H:%M:%S')}] {nome} {bb} s{s} K={K}: densa {rd:.4f} (kappa {sd['cobertura_mediana']:.3f})"
                      f" | proj {rp:.4f} | NFH {rn:.4f} (kappa {sn['cobertura_mediana']:.3f}, ép {aj['seguro']['epoca']})"
                      f" | NFH-c {rc:.4f} (ép {aj['completo']['epoca']}) | conv {err:.0e}", flush=True)
            np.savez_compressed(DIR_P / f"{nome}__{bb}__s{s}.npz", **partes)
    for s in (seeds if com_nbeats else ()):
        if ("N-BEATS-I", "", "", s) in ja:
            continue
        set_seed(s)
        nb = InstanceNorm(NBeatsI(base["d"], L3, H3, degree=Q3), target_idx=-1)
        t0 = time.time()
        nb, info = train_model(nb, tr, va, s, lam=0.0, cfg=CFG3, dev=dev)
        t_nb = time.time() - t0
        nb.eval()
        with torch.no_grad():
            outs = [nb(Xte[i:i + 1024].to(dev)) for i in range(0, len(Xte), 1024)]
        Pn = torch.cat([o["forecast"].cpu() for o in outs]).double().numpy()
        Tn = torch.cat([o["trend"].cpu() for o in outs]).double().numpy()
        Sn = torch.cat([o["seasonal"].cpu() for o in outs]).double().numpy()
        r, a = metricas(Pn, Y)
        prn, sn = resumo(Pn, 10)
        nat = partes_nativas(Tn, Sn, Pn)
        gravar(nome, {**base, "modelo": "N-BEATS-I", "variante": "", "K": "", "backbone": "N-BEATS-I", "seed": s,
                      "RMSE": r, "MAE": a, "val_RMSE": avaliar(nb, Xv, Yv, dev)[0], "epocas": info["epochs"],
                      "tempo_s": t_nb, "params": n_params(nb), **sn})
        np.savez_compressed(DIR_P / f"{nome}__N-BEATS-I__s{s}.npz", proj_eps_T=prn["eps_T"], proj_eps_S=prn["eps_S"],
                            proj_eps_R=prn["eps_R"], nativo_rho_T=nat["rho_T"], nativo_rho_S=nat["rho_S"])
        if s == 0:
            torch.save(nb.state_dict(), DIR_M / f"{nome}__N-BEATS-I__s0.pt")
            np.save(DIR_Y / f"{nome}__N-BEATS-I__s0.npy", Pn.astype(np.float32))
        print(f"[{time.strftime('%H:%M:%S')}] {nome} N-BEATS-I s{s}: {r:.4f} | {t_nb:.0f} s", flush=True)


def main(nomes):
    import os
    # V3_THREADS: número de threads de CPU (padrão 2); com CUDA_VISIBLE_DEVICES="" o processo roda em CPU.
    torch.set_num_threads(int(os.environ.get("V3_THREADS", "2")))
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for nome in nomes:
        rodar(nome, dev)


if __name__ == "__main__":
    main(sys.argv[1:])
