conda activate CFTTS
export PATH=$PWD/utils:$PATH
export PYTHONPATH=$PWD/hifigan:$PYTHONPATH
chmod +x utils/*
chmod +x hifigan/parallel_wavegan/bin/decode.py
