#!/bin/bash
#PBS -N flair2_full
#PBS -q default
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb:scratch_local=150gb:cl_fobos=true
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o flair2_output_DL.log
#PBS -e flair2_error_DL.log

SING_IMAGE="/auto/brno12-cerit/nfs4/home/strakajk/Singularity/containers/remotesensing_24.12_00.SIF"

CODEDIR="/auto/plzen1/home/klamu/FLAIR-2"
DATADIR="/auto/plzen1/home/klamu/FLAIR-2/flair_2_dataset"

singularity exec "$SING_IMAGE" pip install numpy==1.26.4 wandb

export WANDB_API_KEY="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"

cd "$CODEDIR"

echo "Kop�ruji data na scratch disk..."
cp -r "$DATADIR"/* "$SCRATCHDIR/"

# Novy config
sed "s|\./flair_2_dataset|$SCRATCHDIR|g" configsBP/flair-2-config.yml > config_scratch.yml

# tTady Pythonu p�ed�me ten vygenerovan� config_scratch.yml
singularity exec --nv \
    -B "$CODEDIR" \
    -B "$SCRATCHDIR" \
    "$SING_IMAGE" \
    python3 main.py --config_file config_scratch.yml

clean_scratch
