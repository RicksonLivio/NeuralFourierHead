"""Baixa e padroniza 3 conjuntos horarios NOVOS para o experimento de backbones da NFH.

Rodada de 21/11/2026. Saida em experimento_backbones_nfh/dados/<nome>.csv
(coluna `date` ISO 8601, alvo na ultima coluna, agregacao horaria por media, faltantes vazios)
e entradas em catalogo.json no mesmo formato de teste_exemplificacao_nfh/dados/catalogo.json.

Conjuntos (regra de escolha declarada ANTES de olhar a forca do ciclo):
  1. melbourne_pedestrian_southbank: contagem de pedestres do sensor SouthB_T (location_id 35),
     City of Melbourne Open Data, CC BY. Janela fixa 2024-11-22 a 2026-10-31 (a API publica so os
     ultimos ~2 anos; o recorte fixo e o CSV gravado tornam o resultado reproduzivel).
  2. electricity_mt156: consumo (kW, media horaria) do primeiro cliente, na ordem do arquivo, sem nenhuma
     hora zero em 2011-2014 (MT_156), UCI ElectricityLoadDiagrams20112014, CC BY 4.0.
  3. opsd_de_solar: geracao solar real da Alemanha (MW), Open Power System Data, pacote
     time_series 2020-10-06, CC BY 4.0. Covariaveis: carga e geracao eolica. Recorte 2015-2019.

A forca do ciclo diario e medida SO nos 70% iniciais (treino), como em
teste_exemplificacao_nfh/dados/baixar_dados.py (funcao copiada, nao importada).

Uso: .venv/bin/python experimento_backbones_nfh/dados/baixar_dados.py
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
CACHE = Path("/tmp/nfh_backbones_cache")
CACHE.mkdir(exist_ok=True)

URL_MELB = ("https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/"
            "pedestrian-counting-system-monthly-counts-per-hour/exports/csv"
            "?where=location_id%3D35&delimiter=%2C")
URL_ELEC = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
URL_OPSD = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
CORTES = [0.7, 0.15]


def baixar(url: str, destino: Path) -> Path:
    if destino.exists() and destino.stat().st_size > 0:
        return destino
    print(f"baixando {url}", flush=True)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (pesquisa NFH)"})
    with urlopen(req, timeout=600) as r, open(destino, "wb") as f:
        while True:
            bloco = r.read(1 << 20)
            if not bloco:
                break
            f.write(bloco)
    return destino


def forca_ciclo_diario(s: pd.Series) -> dict:
    """Copia de teste_exemplificacao_nfh/dados/baixar_dados.py::compute_daily_cycle_strength (freq horaria)."""
    s = s.dropna()
    if len(s) < 48:
        return {"r2_perfil": None, "acf_12h": None, "acf_24h": None}
    tend = s.rolling(window=24, center=True, min_periods=12).mean()
    res = (s - tend).dropna()
    horas = res.index.hour
    perfil = pd.Series(res.values, index=horas).groupby(level=0).mean()
    pv = perfil.reindex(horas).values
    ss_res = np.sum((res.values - pv) ** 2)
    ss_tot = np.sum((res.values - res.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def acf(x, lag):
        m, v = x.mean(), x.var()
        return float(np.mean((x[:-lag] - m) * (x[lag:] - m)) / v) if v > 0 else None

    return {"r2_perfil": round(float(r2), 4), "acf_12h": round(acf(res.values, 12), 4),
            "acf_24h": round(acf(res.values, 24), 4)}


def ciclo_bruto(s: pd.Series) -> dict:
    """Complemento: SEM remover a media movel de 24 h (que apaga o proprio ciclo diario em series como a solar).
    r2_perfil_bruto = variancia da serie explicada pelo perfil medio por hora do dia (treino);
    acf_24h_bruto = autocorrelacao da serie centrada no lag 24."""
    s = s.dropna()
    perfil = s.groupby(s.index.hour).mean()
    pv = perfil.reindex(s.index.hour).values
    r2 = 1 - np.sum((s.values - pv) ** 2) / np.sum((s.values - s.mean()) ** 2)
    x = s.values - s.mean()
    a24 = float(np.mean(x[:-24] * x[24:]) / x.var())
    return {"r2_perfil_bruto": round(float(r2), 4), "acf_24h_bruto": round(a24, 4)}


def periodos_dominantes(s: pd.Series, n: int = 3) -> list:
    """Periodos (h) dos maiores picos do periodograma do treino, entre 6 h e 180 h (maximos locais) (interpola faltantes)."""
    x = s.interpolate(limit_direction="both")
    x = (x - x.rolling(337, center=True, min_periods=1).mean()).values.astype(float)  # tira a tendencia lenta (anual) antes
    N = len(x)
    p = np.abs(np.fft.rfft(x * np.hanning(N))) ** 2
    f = np.fft.rfftfreq(N, d=1.0)
    per = np.where(f > 0, 1.0 / np.maximum(f, 1e-12), np.inf)
    ok = (per >= 6) & (per <= 180)
    ok[1:-1] &= (p[1:-1] >= p[:-2]) & (p[1:-1] >= p[2:])  # so maximos locais
    idx = np.where(ok)[0]
    ordem = idx[np.argsort(p[idx])[::-1]]
    achados = []
    for i in ordem:
        pr = per[i]
        if all(abs(pr - a) / a > 0.1 for a in achados):
            achados.append(float(pr))
        if len(achados) == n:
            break
    return [int(round(a)) for a in achados]


def r2_janela_TS(s: pd.Series, L: int = 96, H: int = 48, q: int = 2, K: int = 10, passo: int = 24) -> dict:
    """Quanto de uma janela de H=48 h cabe em T (+) S: R2 de minimos quadrados nas colunas
    [1, t, t^2] e cos/sin(2*pi*k*t/H), k=1..K, usando janelas do treino (padronizado, interpolado).
    Mede o teto de cobertura (1 - eps_R) que uma cabeca T+S teria SE o futuro fosse conhecido."""
    x = s.interpolate(limit_direction="both").values.astype(float)
    x = (x - x.mean()) / x.std()
    t = np.arange(H)
    tn = (t - (H - 1) / 2) / H
    cols = [tn ** j for j in range(q + 1)]
    for k in range(1, K + 1):
        cols += [np.cos(2 * np.pi * k * t / H), np.sin(2 * np.pi * k * t / H)]
    B = np.stack(cols, 1)
    P = B @ np.linalg.pinv(B)
    r2s = []
    for i in range(0, len(x) - H, passo):
        y = x[i:i + H]
        yc = y - y.mean()
        den = float(yc @ yc)
        if den < 1e-9:
            continue
        r = y - P @ y
        r2s.append(1 - float(r @ r) / den)
    r2s = np.array(r2s)
    return {"r2_TS_K10_media": round(float(r2s.mean()), 4), "r2_TS_K10_mediana": round(float(np.median(r2s)), 4)}


def finalizar(df: pd.DataFrame, nome: str, fonte_url: str, licenca: str, alvo: str, notas: str) -> dict:
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(idx)
    df.index.name = "date"
    cov = [c for c in df.columns if c != alvo]
    df = df[cov + [alvo]]
    df.to_csv(AQUI / f"{nome}.csv", date_format="%Y-%m-%d %H:%M:%S", float_format="%.6g")
    n = len(df)
    ntr = int(n * CORTES[0])
    treino = df[alvo].iloc[:ntr]
    ciclo = forca_ciclo_diario(treino)
    ciclo.update(ciclo_bruto(treino))
    ciclo.update(r2_janela_TS(treino))
    entrada = {
        "nome": nome, "arquivo": f"{nome}.csv", "fonte_url": fonte_url, "licenca": licenca,
        "freq": "H", "alvo": alvo, "covariaveis": cov,
        "periodos_dominantes_h": periodos_dominantes(treino, 3),
        "cortes": CORTES, "n_obs": int(n),
        "faltantes_frac": round(float(df[alvo].isna().mean()), 4),
        "ciclo_diario": ciclo, "origem": "rodada 2026-11-21 (pesquisa_backbones)",
        "periodo": f"{df.index.min():%Y-%m-%d %H:%M} a {df.index.max():%Y-%m-%d %H:%M}",
        "notas": notas,
    }
    tam = (AQUI / f"{nome}.csv").stat().st_size / 1e6
    print(f"{nome}: {n} h, faltantes {entrada['faltantes_frac']}, {tam:.2f} MB, ciclo {ciclo}, periodos {entrada['periodos_dominantes_h']}")
    return entrada


def melbourne() -> dict:
    p = baixar(URL_MELB, CACHE / "melb_loc35.csv")
    d = pd.read_csv(p, sep=None, engine="python", encoding="utf-8-sig")
    d["date"] = pd.to_datetime(d["sensing_date"]) + pd.to_timedelta(d["hourday"], unit="h")
    d = d[(d["date"] >= "2024-11-22") & (d["date"] < "2026-11-01")]
    s = d.groupby("date")["pedestriancount"].mean().rename("pedestriancount")
    return finalizar(s.to_frame(), "melbourne_pedestrian_southbank", URL_MELB.split("?")[0] + "?where=location_id%3D35",
                     "CC BY 4.0 (City of Melbourne Open Data; campo license da API: 'CC BY')", "pedestriancount",
                     "Pedestrian Counting System, sensor SouthB_T (location_id 35), contagem por hora, hora local de Melbourne. "
                     "Janela fixa 2024-11-22 a 2026-10-31 (a API so mantem cerca de 2 anos). Sem covariaveis (direction_1/2 somam o alvo). "
                     "Ciclo diario e semanal fortes; ha eventos (feriados, jogos).")


def eletricidade() -> dict:
    zp = baixar(URL_ELEC, CACHE / "elec.zip")
    with zipfile.ZipFile(zp) as z:
        with z.open("LD2011_2014.txt") as f:
            d = pd.read_csv(f, sep=";", decimal=",", index_col=0, parse_dates=True)
    h = d.resample("1h").mean()
    sem_zero = [c for c in h.columns if (h[c] == 0).mean() == 0 and h[c].notna().all()]
    cliente = sem_zero[0]  # regra declarada: primeiro na ordem do arquivo
    print(f"electricity: {len(sem_zero)} clientes sem hora zero; escolhido {cliente}")
    s = h[cliente].rename("MT_kW")
    # o arquivo marca o fim do intervalo de 15 min; a media horaria rotula o inicio da hora
    return finalizar(s.to_frame(), "electricity_mt156", "https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014",
                     "CC BY 4.0 (UCI ElectricityLoadDiagrams20112014)", "MT_kW",
                     f"Cliente {cliente} (primeiro, na ordem do arquivo, sem hora zero em 2011-2014), kW medio por hora "
                     "(15 min agregado por media). Sem covariaveis. Cliente fixado por regra, nao pela forca do ciclo.")


def opsd() -> dict:
    p = baixar(URL_OPSD, CACHE / "opsd60.csv")
    cols = ["utc_timestamp", "DE_load_actual_entsoe_transparency", "DE_wind_generation_actual", "DE_solar_generation_actual"]
    d = pd.read_csv(p, usecols=cols, parse_dates=["utc_timestamp"])
    d["date"] = d["utc_timestamp"].dt.tz_convert(None)
    d = d.set_index("date").drop(columns="utc_timestamp")
    d = d[cols[1:]]  # usecols NAO preserva a ordem pedida: reordenar por nome antes de renomear
    d = d.loc["2015-01-01":"2019-12-31 23:00"]
    d.columns = ["load_MW", "wind_MW", "solar_MW"]
    return finalizar(d, "opsd_de_solar", "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv",
                     "licenca dos dados nao declarada pelo OPSD (time_series 2020-10-06; dados primarios ENTSO-E); ver DATA.md do repositorio publico", "solar_MW",
                     "Geracao solar real da Alemanha (DE_solar_generation_actual), horario em UTC, 2015-2019. Covariaveis: carga real "
                     "(DE_load_actual_entsoe_transparency) e geracao eolica (DE_wind_generation_actual). Zero a noite; amplitude "
                     "varia com a estacao e cresce com a capacidade instalada.")


def main():
    novos = [melbourne(), eletricidade(), opsd()]
    cat_path = AQUI / "catalogo.json"
    cat = json.loads(cat_path.read_text(encoding="utf-8")) if cat_path.exists() else []
    nomes = {e["nome"] for e in novos}
    cat = [e for e in cat if e["nome"] not in nomes and e["nome"] != "electricity_mt124"] + novos  # mt124: nome de uma 1a versao
    cat_path.write_text(json.dumps(cat, ensure_ascii=False, indent=2), encoding="utf-8")
    print("catalogo.json atualizado:", [e["nome"] for e in cat])


if __name__ == "__main__":
    sys.exit(main())
