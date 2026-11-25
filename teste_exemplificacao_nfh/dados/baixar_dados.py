#!/usr/bin/env python3
"""
Baixar e processar datasets para o teste de exemplificação da NFH.

Este script:
1. Baixa datasets públicos de fontes verificadas (UCI, ETT repository, TensorFlow)
2. Processa datasets já existentes no repositório da dissertação
3. Converte todos para formato padronizado (horário, date ISO 8601, alvo última coluna)
4. Calcula métricas de força do ciclo diário (R² do perfil por hora, ACF em 12h e 24h)
5. Gera o catalogo.json com metadados completos

Datasets processados:
- ETTh1, ETTh2: Electricity Transformer Temperature (Zhou et al., 2021)
- Jena Climate 2009-2016: Temperatura (MPI-BGC / TensorFlow)
- Air Quality UCI: Benzeno e poluentes (De Vito et al., 2008)
- Bike Sharing UCI: Aluguel de bicicletas Washington D.C. (Fanaee-T & Gama, 2014)
- Seoul Bike Sharing UCI: Aluguel de bicicletas Seoul (2020)
- Household Electric Power UCI: Consumo doméstico (Hebrail & Berard, 2006)

Autor: Agente PESQUISA_DADOS
Data: 2026-11-16
"""

import os
import sys
import json
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional


# Configuração de caminhos
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
DATA_V3 = REPO_ROOT / "data" / "v3_bruto"
DATA_CONTROLE = REPO_ROOT / "data" / "controle"
OUTPUT_DIR = SCRIPT_DIR


# URLs de download verificados em 2026-11-16
URLS = {
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "jena_climate": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
    "air_quality": "https://archive.ics.uci.edu/static/public/360/air+quality.zip",
    "bike_sharing": "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
    "seoul_bike": "https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip",
    "household_power": "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
}


def download_file(url: str, dest_path: Path, skip_if_exists: bool = True) -> Path:
    """Baixa um arquivo de uma URL se não existir localmente."""
    if skip_if_exists and dest_path.exists():
        print(f"  [SKIP] {dest_path.name} já existe")
        return dest_path
    
    print(f"  [DOWNLOAD] {url}")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"  [OK] Salvo em {dest_path.name} ({dest_path.stat().st_size / 1024:.1f} KB)")
        return dest_path
    except Exception as e:
        print(f"  [ERRO] Falha no download: {e}")
        raise


def extract_zip(zip_path: Path, extract_to: Path, file_pattern: Optional[str] = None) -> Path:
    """Extrai um arquivo específico de um ZIP."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()
        if file_pattern:
            target = [n for n in names if file_pattern in n.lower()]
            if not target:
                raise ValueError(f"Padrão '{file_pattern}' não encontrado em {names}")
            target = target[0]
        else:
            target = names[0]
        
        extracted = extract_to / Path(target).name
        with zf.open(target) as src, open(extracted, 'wb') as dst:
            dst.write(src.read())
        return extracted


def compute_daily_cycle_strength(series: pd.Series, freq_hours: int = 1) -> Dict[str, float]:
    """
    Calcula a força do ciclo diário de uma série temporal.
    
    Método:
    1. Remove tendência via média móvel centrada de 24h
    2. Calcula o perfil médio por hora do dia
    3. Mede R² do perfil sobre o resíduo
    4. Calcula ACF do resíduo nos lags 12 e 24
    
    Retorna dict com r2_perfil, acf_12h, acf_24h.
    """
    # Remove NaNs para cálculo
    s = series.dropna()
    if len(s) < 48:  # mínimo 2 dias
        return {"r2_perfil": np.nan, "acf_12h": np.nan, "acf_24h": np.nan}
    
    # Média móvel centrada de 24h (remove tendência de baixa frequência)
    window = 24 // freq_hours
    trend = s.rolling(window=window, center=True, min_periods=window // 2).mean()
    residual = s - trend
    residual = residual.dropna()
    
    if len(residual) < 24:
        return {"r2_perfil": np.nan, "acf_12h": np.nan, "acf_24h": np.nan}
    
    # Perfil médio por hora do dia
    if hasattr(residual.index, 'hour'):
        hours = residual.index.hour
    else:
        hours = np.arange(len(residual)) % 24
    
    # Calcula média por hora
    hourly_profile = pd.Series(residual.values, index=hours).groupby(level=0).mean()
    profile_values = hourly_profile.reindex(hours).values
    
    # R² do perfil (variância explicada pelo ciclo diário)
    ss_res = np.sum((residual.values - profile_values) ** 2)
    ss_tot = np.sum((residual.values - residual.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # ACF nos lags 12 e 24 horas
    def acf(x, lag):
        n = len(x)
        if lag >= n:
            return np.nan
        mean = np.mean(x)
        var = np.var(x)
        if var == 0:
            return np.nan
        cov = np.mean((x[:-lag] - mean) * (x[lag:] - mean))
        return cov / var
    
    acf_12 = acf(residual.values, 12 // freq_hours)
    acf_24 = acf(residual.values, 24 // freq_hours)
    
    return {
        "r2_perfil": round(r2, 4),
        "acf_12h": round(acf_12, 4) if not np.isnan(acf_12) else None,
        "acf_24h": round(acf_24, 4) if not np.isnan(acf_24) else None,
    }


def process_etth(source_path: Path, name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Processa ETTh1 ou ETTh2 para o formato padronizado."""
    print(f"\nProcessando {name}...")
    df = pd.read_csv(source_path)
    
    # Já está em formato horário com date e OT como alvo
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    # Ordem: covariáveis primeiro, alvo (OT) por último
    cols = [c for c in df.columns if c != 'OT'] + ['OT']
    df = df[cols]
    
    # Calcula força do ciclo diário apenas no treino (70%)
    n_train = int(len(df) * 0.7)
    train_target = df['OT'].iloc[:n_train]
    train_target.index = df.index[:n_train]
    cycle_strength = compute_daily_cycle_strength(train_target)
    
    meta = {
        "nome": name,
        "arquivo": f"{name}.csv",
        "fonte_url": URLS[name],
        "licenca": "CC BY-ND 4.0 (ETDataset)",
        "freq": "H",
        "alvo": "OT",
        "covariaveis": [c for c in df.columns if c != 'OT'],
        "periodos_dominantes_h": [24, 168],
        "cortes": [0.7, 0.15],
        "n_obs": len(df),
        "faltantes_frac": 0.0,
        "ciclo_diario": cycle_strength,
        "origem": "dissertação",
        "notas": f"Electricity Transformer Temperature. Alvo: temperatura do óleo (OT). Ciclo diário {'forte' if cycle_strength['r2_perfil'] > 0.4 else 'moderado'} (R²={cycle_strength['r2_perfil']:.3f})."
    }
    
    return df, meta


def process_jena_climate(zip_path: Path, recorte_anos: Tuple[int, int] = (2009, 2011)) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Processa Jena Climate 2009-2016 para formato horário.
    
    Procedimento:
    1. Extrai CSV do ZIP
    2. Trata timestamps duplicados (remove) e wv/max.wv = -9999 (substitui por 0)
    3. Converte direção do vento (wd) para vetor (Wx, Wy)
    4. Agrega para horário usando média (resample right-closed, right-labeled)
    5. Recorta período especificado
    6. Seleciona variáveis: T (alvo), p, rh, Wx, Wy (covariáveis principais)
    """
    print("\nProcessando Jena Climate...")
    
    # Extrai CSV
    temp_dir = OUTPUT_DIR / "temp"
    temp_dir.mkdir(exist_ok=True)
    csv_path = extract_zip(zip_path, temp_dir, "jena_climate")
    
    # Carrega dados
    df = pd.read_csv(csv_path)
    print(f"  Linhas originais: {len(df)}")
    
    # Parse datetime
    df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
    
    # Remove duplicatas de timestamp (mantém primeira ocorrência)
    n_before = len(df)
    df = df.drop_duplicates(subset=['Date Time'], keep='first')
    print(f"  Após remover duplicatas: {len(df)} (removidas: {n_before - len(df)})")
    
    # Trata valores -9999 em wv e max. wv
    for col in ['wv (m/s)', 'max. wv (m/s)']:
        if col in df.columns:
            n_bad = (df[col] == -9999).sum()
            if n_bad > 0:
                print(f"  Substituindo {n_bad} valores -9999 em {col} por 0")
                df.loc[df[col] == -9999, col] = 0
    
    # Converte direção do vento para vetor
    wd_rad = np.deg2rad(df['wd (deg)'].values)
    wv = df['wv (m/s)'].values
    df['Wx'] = wv * np.cos(wd_rad)
    df['Wy'] = wv * np.sin(wd_rad)
    
    # Set index e resample para horário
    df = df.set_index('Date Time')
    df_hourly = df.resample('1h', closed='right', label='right').mean()
    print(f"  Após agregação horária: {len(df_hourly)}")
    
    # Recorta período (anos completos)
    start = f"{recorte_anos[0]}-01-01 01:00:00"
    end = f"{recorte_anos[1]+1}-01-01 00:00:00"
    df_hourly = df_hourly.loc[start:end]
    print(f"  Após recorte {recorte_anos[0]}-{recorte_anos[1]}: {len(df_hourly)}")
    
    # Verifica faltantes
    n_missing = df_hourly.isna().any(axis=1).sum()
    print(f"  Horas com dados faltantes: {n_missing}")
    
    # Seleciona variáveis para versão principal (M)
    # Covariáveis: p (pressão), rh (umidade), Wx, Wy (vento)
    # Alvo: T (temperatura)
    cols_keep = ['p (mbar)', 'rh (%)', 'Wx', 'Wy', 'T (degC)']
    df_out = df_hourly[cols_keep].copy()
    df_out.columns = ['p_mbar', 'rh_pct', 'Wx', 'Wy', 'T_degC']
    
    # Reset index para coluna date
    df_out = df_out.reset_index()
    df_out = df_out.rename(columns={'Date Time': 'date'})
    df_out = df_out.set_index('date')
    
    # Calcula força do ciclo diário apenas no treino
    n_train = int(len(df_out) * 0.7)
    train_target = df_out['T_degC'].iloc[:n_train]
    train_target.index = df_out.index[:n_train]
    cycle_strength = compute_daily_cycle_strength(train_target)
    
    meta = {
        "nome": "jena_climate",
        "arquivo": "jena_climate.csv",
        "fonte_url": URLS["jena_climate"],
        "licenca": "CC BY 4.0 (MPI-BGC)",
        "freq": "H",
        "alvo": "T_degC",
        "covariaveis": ["p_mbar", "rh_pct", "Wx", "Wy"],
        "periodos_dominantes_h": [24, 8766],  # diário e anual
        "cortes": [0.7, 0.15],
        "n_obs": len(df_out),
        "faltantes_frac": round(n_missing / len(df_out), 4),
        "ciclo_diario": cycle_strength,
        "origem": "dissertação",
        "recorte": f"{recorte_anos[0]}-{recorte_anos[1]}",
        "notas": f"Jena Climate (MPI-BGC/TensorFlow). Alvo: temperatura do ar. Agregado de 10min para horário. Recorte: {recorte_anos[0]}-{recorte_anos[1]}. Ciclo diário forte (R²={cycle_strength['r2_perfil']:.3f})."
    }
    
    # Limpa temp
    csv_path.unlink()
    
    return df_out, meta


def process_air_quality(zip_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Processa UCI Air Quality para formato padronizado.
    
    Alvo: C6H6(GT) (benzeno)
    Covariáveis: PT08.S1(CO), PT08.S3(NOx), PT08.S4(NO2), PT08.S5(O3), T, RH, AH
    Exclui: PT08.S2(NMHC) por ser quase função de C6H6, NMHC(GT)/CO(GT)/NOx(GT)/NO2(GT) por >17% faltantes
    """
    print("\nProcessando Air Quality UCI...")
    
    # Extrai CSV
    temp_dir = OUTPUT_DIR / "temp"
    temp_dir.mkdir(exist_ok=True)
    csv_path = extract_zip(zip_path, temp_dir, "airqualityuci.csv")
    
    # Carrega com separador ; e decimal ,
    df = pd.read_csv(csv_path, sep=';', decimal=',')
    
    # Remove colunas vazias no final
    df = df.dropna(axis=1, how='all')
    
    # Remove linhas completamente vazias
    df = df.dropna(axis=0, how='all')
    print(f"  Linhas após limpeza: {len(df)}")
    
    # Cria datetime
    df['date'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
    
    # Substitui -200 por NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df.loc[df[col] == -200, col] = np.nan
    
    # Seleciona colunas (exclui S2 que é quase duplicata de C6H6)
    cols_final = ['date', 'PT08.S1(CO)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 
                  'PT08.S5(O3)', 'T', 'RH', 'AH', 'C6H6(GT)']
    df_out = df[cols_final].copy()
    
    # Renomeia para nomes mais limpos
    df_out.columns = ['date', 'S1_CO', 'S3_NOx', 'S4_NO2', 'S5_O3', 'T', 'RH', 'AH', 'C6H6']
    
    df_out = df_out.set_index('date')
    
    # Conta faltantes
    n_missing_target = df_out['C6H6'].isna().sum()
    faltantes_frac = df_out.isna().any(axis=1).sum() / len(df_out)
    print(f"  Faltantes no alvo C6H6: {n_missing_target} ({n_missing_target/len(df_out)*100:.2f}%)")
    print(f"  Fração de linhas com algum faltante: {faltantes_frac:.4f}")
    
    # Calcula força do ciclo diário apenas no treino
    n_train = int(len(df_out) * 0.7)
    train_target = df_out['C6H6'].iloc[:n_train].dropna()
    if len(train_target) > 48:
        cycle_strength = compute_daily_cycle_strength(train_target)
    else:
        cycle_strength = {"r2_perfil": None, "acf_12h": None, "acf_24h": None}
    
    meta = {
        "nome": "air_quality",
        "arquivo": "air_quality.csv",
        "fonte_url": URLS["air_quality"],
        "licenca": "CC BY 4.0 (UCI)",
        "freq": "H",
        "alvo": "C6H6",
        "covariaveis": ["S1_CO", "S3_NOx", "S4_NO2", "S5_O3", "T", "RH", "AH"],
        "periodos_dominantes_h": [24, 168],  # diário e semanal (tráfego)
        "cortes": [0.7, 0.15],
        "n_obs": len(df_out),
        "faltantes_frac": round(faltantes_frac, 4),
        "ciclo_diario": cycle_strength,
        "origem": "dissertação",
        "notas": f"UCI Air Quality (De Vito et al., 2008). Alvo: benzeno C6H6(GT). Ressalva: C6H6 é quase função monótona de S2(NMHC), excluído. Ciclo diário moderado (R²={cycle_strength['r2_perfil']:.3f} se disponível)."
    }
    
    # Limpa temp
    csv_path.unlink()
    
    return df_out, meta


def process_bike_sharing(download_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Processa UCI Bike Sharing (Washington D.C.) para formato padronizado.
    
    Alvo: cnt (contagem total de aluguéis)
    Covariáveis: temp, atemp, hum, windspeed, weathersit, holiday, workingday
    """
    print("\nProcessando UCI Bike Sharing...")
    
    # Download
    zip_path = download_dir / "bike_sharing.zip"
    download_file(URLS["bike_sharing"], zip_path)
    
    # Extrai hour.csv
    csv_path = extract_zip(zip_path, download_dir, "hour.csv")
    
    df = pd.read_csv(csv_path)
    print(f"  Linhas originais: {len(df)}")
    
    # Cria datetime completo
    df['date'] = pd.to_datetime(df['dteday']) + pd.to_timedelta(df['hr'], unit='h')
    
    # Seleciona covariáveis e alvo
    # Exclui casual e registered (somam ao alvo)
    cols = ['date', 'temp', 'atemp', 'hum', 'windspeed', 'weathersit', 'holiday', 'workingday', 'cnt']
    df_out = df[cols].copy()
    df_out = df_out.set_index('date')
    
    # Verifica lacunas na série temporal
    full_range = pd.date_range(df_out.index.min(), df_out.index.max(), freq='h')
    missing_hours = len(full_range) - len(df_out)
    print(f"  Horas ausentes na grade temporal: {missing_hours}")
    
    # Calcula força do ciclo diário
    n_train = int(len(df_out) * 0.7)
    train_target = df_out['cnt'].iloc[:n_train]
    train_target.index = df_out.index[:n_train]
    cycle_strength = compute_daily_cycle_strength(train_target)
    
    meta = {
        "nome": "bike_sharing_dc",
        "arquivo": "bike_sharing_dc.csv",
        "fonte_url": URLS["bike_sharing"],
        "licenca": "CC BY 4.0 (UCI)",
        "freq": "H",
        "alvo": "cnt",
        "covariaveis": ["temp", "atemp", "hum", "windspeed", "weathersit", "holiday", "workingday"],
        "periodos_dominantes_h": [24, 168],  # diário forte + semanal forte
        "cortes": [0.7, 0.15],
        "n_obs": len(df_out),
        "faltantes_frac": round(missing_hours / len(full_range), 4),
        "ciclo_diario": cycle_strength,
        "origem": "novo",
        "notas": f"UCI Bike Sharing (Capital Bikeshare, Washington D.C., 2011-2012). Alvo: cnt (total de aluguéis). Ciclo diário muito forte (R²={cycle_strength['r2_perfil']:.3f}), picos de commuting."
    }
    
    # Limpa
    csv_path.unlink()
    zip_path.unlink()
    
    return df_out, meta


def process_seoul_bike(download_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Processa UCI Seoul Bike Sharing para formato padronizado.
    
    Alvo: Rented Bike Count
    Covariáveis: Temperature, Humidity, Wind speed, Visibility, Dew point temperature, Solar Radiation, Rainfall, Snowfall
    """
    print("\nProcessando UCI Seoul Bike Sharing...")
    
    # Download
    zip_path = download_dir / "seoul_bike.zip"
    download_file(URLS["seoul_bike"], zip_path)
    
    # Extrai CSV
    csv_path = extract_zip(zip_path, download_dir, "seoulbikedata.csv")
    
    # Tenta diferentes encodings
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            break
        except:
            continue
    
    print(f"  Linhas: {len(df)}")
    print(f"  Colunas: {list(df.columns)}")
    
    # Cria datetime
    df['date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y') + pd.to_timedelta(df['Hour'], unit='h')
    
    # Seleciona colunas (adapta aos nomes reais)
    # Nomes podem variar, vamos verificar
    possible_target_names = ['Rented Bike Count', 'Rented Bike count', 'bike_count', 'count']
    target_col = None
    for name in possible_target_names:
        if name in df.columns:
            target_col = name
            break
    
    if target_col is None:
        print(f"  AVISO: Coluna alvo não encontrada. Colunas disponíveis: {list(df.columns)}")
        target_col = df.columns[1]  # Segunda coluna após Date
    
    # Mapeia nomes de colunas
    col_mapping = {
        'Temperature(°C)': 'Temperature',
        'Temperature(.C)': 'Temperature',
        'Humidity(%)': 'Humidity',
        'Wind speed (m/s)': 'WindSpeed',
        'Windspeed (m/s)': 'WindSpeed',
        'Visibility (10m)': 'Visibility',
        'Visibility(10m)': 'Visibility',
        'Dew point temperature(°C)': 'DewPoint',
        'Dew point temperature(.C)': 'DewPoint',
        'Solar Radiation (MJ/m2)': 'SolarRadiation',
        'Solar Radiation(MJ/m2)': 'SolarRadiation',
        'Rainfall(mm)': 'Rainfall',
        'Rainfall (mm)': 'Rainfall',
        'Snowfall (cm)': 'Snowfall',
        'Snowfall(cm)': 'Snowfall',
    }
    
    df = df.rename(columns=col_mapping)
    df = df.rename(columns={target_col: 'RentedBikeCount'})
    
    # Covariáveis disponíveis
    covars = ['Temperature', 'Humidity', 'WindSpeed', 'Visibility', 'DewPoint', 
              'SolarRadiation', 'Rainfall', 'Snowfall']
    covars_available = [c for c in covars if c in df.columns]
    
    cols_final = ['date'] + covars_available + ['RentedBikeCount']
    df_out = df[cols_final].copy()
    df_out = df_out.set_index('date')
    
    # Calcula força do ciclo diário
    n_train = int(len(df_out) * 0.7)
    train_target = df_out['RentedBikeCount'].iloc[:n_train]
    train_target.index = df_out.index[:n_train]
    cycle_strength = compute_daily_cycle_strength(train_target)
    
    meta = {
        "nome": "seoul_bike",
        "arquivo": "seoul_bike.csv",
        "fonte_url": URLS["seoul_bike"],
        "licenca": "CC BY 4.0 (UCI)",
        "freq": "H",
        "alvo": "RentedBikeCount",
        "covariaveis": covars_available,
        "periodos_dominantes_h": [24, 168],
        "cortes": [0.7, 0.15],
        "n_obs": len(df_out),
        "faltantes_frac": 0.0,
        "ciclo_diario": cycle_strength,
        "origem": "novo",
        "notas": f"UCI Seoul Bike Sharing (Seoul, 2017-2018). Um ano completo (8760h). Alvo: contagem de aluguéis. Ciclo diário forte (R²={cycle_strength['r2_perfil']:.3f})."
    }
    
    # Limpa
    csv_path.unlink()
    zip_path.unlink()
    
    return df_out, meta


def process_household_power(download_dir: Path, recorte_anos: Tuple[int, int] = (2007, 2009)) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Processa UCI Household Electric Power Consumption para formato horário.
    
    Alvo: Global_active_power (consumo ativo global em kW)
    Covariáveis: Voltage, Global_intensity, Sub_metering_1, Sub_metering_2, Sub_metering_3
    
    Agregação: média horária de dados em 1 minuto.
    Recorte: período especificado para manter arquivo < 60 MB.
    """
    print("\nProcessando UCI Household Electric Power Consumption...")
    
    # Download (arquivo grande ~20MB comprimido, ~127MB descomprimido)
    zip_path = download_dir / "household_power.zip"
    download_file(URLS["household_power"], zip_path)
    
    # Extrai (pode demorar)
    print("  Extraindo ZIP (arquivo grande)...")
    temp_dir = download_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Procura o arquivo txt
        txt_files = [n for n in zf.namelist() if n.endswith('.txt')]
        if txt_files:
            target = txt_files[0]
            csv_path = temp_dir / "household_power.txt"
            with zf.open(target) as src, open(csv_path, 'wb') as dst:
                dst.write(src.read())
        else:
            raise ValueError(f"Nenhum arquivo .txt encontrado no ZIP")
    
    print(f"  Carregando dados (pode demorar)...")
    # Carrega com separador ; e trata ? como NA
    df = pd.read_csv(csv_path, sep=';', na_values=['?'], low_memory=False)
    print(f"  Linhas originais: {len(df)}")
    
    # Cria datetime
    df['date'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df = df.set_index('date')
    
    # Recorta período antes de agregar (economiza memória)
    start = f"{recorte_anos[0]}-01-01"
    end = f"{recorte_anos[1]}-12-31 23:59:59"
    df = df.loc[start:end]
    print(f"  Após recorte {recorte_anos[0]}-{recorte_anos[1]}: {len(df)}")
    
    # Converte colunas numéricas
    numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Agrega para horário (média)
    print("  Agregando para horário...")
    df_hourly = df[numeric_cols].resample('1h').mean()
    print(f"  Após agregação: {len(df_hourly)}")
    
    # Conta faltantes
    n_missing = df_hourly.isna().any(axis=1).sum()
    faltantes_frac = n_missing / len(df_hourly)
    print(f"  Horas com dados faltantes: {n_missing} ({faltantes_frac*100:.2f}%)")
    
    # Seleciona colunas: covariáveis primeiro, alvo por último
    cols_final = ['Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 
                  'Sub_metering_3', 'Global_active_power']
    df_out = df_hourly[cols_final]
    
    # Calcula força do ciclo diário
    n_train = int(len(df_out) * 0.7)
    train_target = df_out['Global_active_power'].iloc[:n_train].dropna()
    if len(train_target) > 48:
        cycle_strength = compute_daily_cycle_strength(train_target)
    else:
        cycle_strength = {"r2_perfil": None, "acf_12h": None, "acf_24h": None}
    
    meta = {
        "nome": "household_power",
        "arquivo": "household_power.csv",
        "fonte_url": URLS["household_power"],
        "licenca": "CC BY 4.0 (UCI)",
        "freq": "H",
        "alvo": "Global_active_power",
        "covariaveis": ["Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"],
        "periodos_dominantes_h": [24, 168],
        "cortes": [0.7, 0.15],
        "n_obs": len(df_out),
        "faltantes_frac": round(faltantes_frac, 4),
        "ciclo_diario": cycle_strength,
        "origem": "novo",
        "recorte": f"{recorte_anos[0]}-{recorte_anos[1]}",
        "notas": f"UCI Household Electric Power (França, 2006-2010). Recorte: {recorte_anos[0]}-{recorte_anos[1]}. Agregado de 1min para horário. Alvo: consumo ativo global (kW). {faltantes_frac*100:.1f}% faltantes."
    }
    
    # Limpa
    csv_path.unlink()
    zip_path.unlink()
    
    return df_out, meta


def main():
    """Função principal que processa todos os datasets."""
    print("=" * 70)
    print("DOWNLOAD E PROCESSAMENTO DE DATASETS PARA TESTE NFH")
    print("=" * 70)
    print(f"Diretório de saída: {OUTPUT_DIR}")
    print(f"Data: {datetime.now().isoformat()}")
    
    catalogo = []
    
    # 1. ETTh1 (do repositório local)
    print("\n" + "=" * 50)
    print("1. DATASETS DA DISSERTAÇÃO (origem: local)")
    print("=" * 50)
    
    etth1_path = DATA_CONTROLE / "ETTh1.csv"
    if etth1_path.exists():
        df, meta = process_etth(etth1_path, "ETTh1")
        df.to_csv(OUTPUT_DIR / "ETTh1.csv")
        catalogo.append(meta)
        print(f"  Salvo: ETTh1.csv ({len(df)} linhas)")
    else:
        print(f"  AVISO: {etth1_path} não encontrado")
    
    # 2. ETTh2 (do repositório local)
    etth2_path = DATA_V3 / "ETTh2.csv"
    if etth2_path.exists():
        df, meta = process_etth(etth2_path, "ETTh2")
        df.to_csv(OUTPUT_DIR / "ETTh2.csv")
        catalogo.append(meta)
        print(f"  Salvo: ETTh2.csv ({len(df)} linhas)")
    else:
        print(f"  AVISO: {etth2_path} não encontrado")
    
    # 3. Jena Climate (do repositório local)
    jena_zip = DATA_V3 / "jena_climate_2009_2016.csv.zip"
    if jena_zip.exists():
        df, meta = process_jena_climate(jena_zip, recorte_anos=(2009, 2011))
        df.to_csv(OUTPUT_DIR / "jena_climate.csv")
        catalogo.append(meta)
        print(f"  Salvo: jena_climate.csv ({len(df)} linhas)")
    else:
        print(f"  AVISO: {jena_zip} não encontrado")
    
    # 4. Air Quality (do repositório local)
    airq_zip = DATA_V3 / "air+quality.zip"
    if airq_zip.exists():
        df, meta = process_air_quality(airq_zip)
        df.to_csv(OUTPUT_DIR / "air_quality.csv")
        catalogo.append(meta)
        print(f"  Salvo: air_quality.csv ({len(df)} linhas)")
    else:
        print(f"  AVISO: {airq_zip} não encontrado")
    
    # 5-7. Novos datasets (download)
    print("\n" + "=" * 50)
    print("2. NOVOS DATASETS (download da internet)")
    print("=" * 50)
    
    # Bike Sharing DC
    try:
        df, meta = process_bike_sharing(OUTPUT_DIR)
        df.to_csv(OUTPUT_DIR / "bike_sharing_dc.csv")
        catalogo.append(meta)
        print(f"  Salvo: bike_sharing_dc.csv ({len(df)} linhas)")
    except Exception as e:
        print(f"  ERRO ao processar Bike Sharing: {e}")
    
    # Seoul Bike
    try:
        df, meta = process_seoul_bike(OUTPUT_DIR)
        df.to_csv(OUTPUT_DIR / "seoul_bike.csv")
        catalogo.append(meta)
        print(f"  Salvo: seoul_bike.csv ({len(df)} linhas)")
    except Exception as e:
        print(f"  ERRO ao processar Seoul Bike: {e}")
    
    # Household Power (recorte 2007-2009 para ~26k horas)
    try:
        df, meta = process_household_power(OUTPUT_DIR, recorte_anos=(2007, 2009))
        df.to_csv(OUTPUT_DIR / "household_power.csv")
        catalogo.append(meta)
        print(f"  Salvo: household_power.csv ({len(df)} linhas)")
    except Exception as e:
        print(f"  ERRO ao processar Household Power: {e}")
    
    # Salva catálogo
    print("\n" + "=" * 50)
    print("SALVANDO CATÁLOGO")
    print("=" * 50)
    
    catalogo_path = OUTPUT_DIR / "catalogo.json"
    with open(catalogo_path, 'w', encoding='utf-8') as f:
        json.dump(catalogo, f, indent=2, ensure_ascii=False)
    print(f"Catálogo salvo: {catalogo_path}")
    
    # Resumo
    print("\n" + "=" * 70)
    print("RESUMO")
    print("=" * 70)
    print(f"Datasets processados: {len(catalogo)}")
    for item in catalogo:
        r2 = item['ciclo_diario'].get('r2_perfil', 'N/A')
        r2_str = f"{r2:.3f}" if isinstance(r2, float) else "N/A"
        print(f"  - {item['nome']}: {item['n_obs']} obs, alvo={item['alvo']}, R²_diário={r2_str}")
    
    # Limpa diretório temp se existir
    temp_dir = OUTPUT_DIR / "temp"
    if temp_dir.exists():
        for f in temp_dir.iterdir():
            f.unlink()
        temp_dir.rmdir()
    
    print("\nConcluído!")
    return catalogo


if __name__ == "__main__":
    catalogo = main()
