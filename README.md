# Segmentace dat dálkového průzkumu Země pomocí neuronových sítí

Repozitář obsahuje zdrojové kódy k bakalářské práci. Jedná se o modifikovaný kód od tvůrců FLAIR-2 (viz: https://github.com/IGNF/FLAIR-2). Kód implementuje sémantickou segmentaci nad datasetem **FLAIR-2** s využitím frameworku PyTorch Lightning.

## 📦 Struktura projektu

* `src/` - složka se zdrojovými kódy (modely, datamoduly, metriky).
* `config/` - konfigurační soubory `.yml` pro jednotlivé experimenty.
* `main.py` - hlavní spouštěcí skript.
* `run.sh` - PBS skript pro spuštění úloh na výpočetním clusteru MetaCentrum.
* `requirements.txt` - seznam závislostí.

## 💾 Data a prostředí

* **Dataset:** FLAIR-2 (data se stahují z oficiálního webu, při běhu na clusteru se kopírují na lokální `$SCRATCHDIR`).
* **Prostředí:** Výpočty jsou přizpůsobeny pro Singularity kontejner `remotesensing_24.12_00.SIF`.

## 🚀 Spuštění projektu

### 1. Instalace závislostí
V případě spuštění mimo kontejner nainstalujte balíčky pomocí:
```bash
pip install -r requirements.txt

### 2. Lokální spuštění
Trénování nebo evaluace se spouští pomocí `main.py` s předáním cesty ke konfiguračnímu souboru přes parametr `--config_file`:
```bash
python main.py --config_file config/flair-2-config.yml

### 3. Spuštění na MetaCentru (PBS)
Úloha se do plánovacího systému odesílá příkazem:
```bash
qsub run.sh

Skript run.sh automaticky alokuje výpočetní uzel s GPU, doinstaluje potřebné knihovny (wandb), překopíruje dataset na lokální scratch, pomocí nástroje sed vygeneruje upravenou konfiguraci config_scratch.yml se správnými cestami k datům a spustí výpočet přes Singularity kontejner

## Logování a sledování
Průběh trénování, hodnoty ztrátové funkce a metriky mIoU se automaticky logují do:
* **TensorBoard** (lokální ukládání logů)
* **Weights & Biases (WandB)** (vzdálená správa experimentů, vyžaduje platný API klíč nastavený v `run.sh`)
