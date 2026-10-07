#!/bin/bash
# [2026-09-28] C-old 5090 mirror env: bit-identical copy of the 4090 C-old runtime (uv CPython 3.13.9, venv site-packages,
# c8old/xvla code, xvla-base + bart-large HF snapshots, legacy c8old_train/val) from one tar written by the 4090 to NFS.
set -e; E=/srv/data/johann/c8old5090; T=/srv/data/johann/c8old5090_tar
want=$(awk '{print $1}' $T/env4090.tar.sha256); got=$(sha256sum $T/env4090.tar | awk '{print $1}')
[ "$want" = "$got" ] && echo "E0 tar sha256 OK $got" || { echo "E0 tar sha256 MISMATCH"; exit 1; }
rm -rf $E/stage $E/py $E/venv $E/xvla $E/hf $E/data/c8old_train $E/data/c8old_val; mkdir -p $E/stage $E/venv/lib/python3.13 $E/venv/bin $E/hf/hub $E/data
tar -xf $T/env4090.tar -C $E/stage
mv $E/stage/cpython-3.13.9-linux-x86_64-gnu $E/py; mv $E/stage/site-packages $E/venv/lib/python3.13/; mv $E/stage/xvla $E/xvla
mv $E/stage/data/c8old_train $E/stage/data/c8old_val $E/stage/data/c8old_provenance.json $E/data/
mv $E/stage/models--lerobot--xvla-base $E/stage/models--facebook--bart-large $E/hf/hub/; cp $E/stage/pyvenv.cfg $E/venv/pyvenv.cfg.4090
sed "s#^home = .*#home = $E/py/bin#" $E/venv/pyvenv.cfg.4090 > $E/venv/pyvenv.cfg; ln -sf $E/py/bin/python3.13 $E/venv/bin/python; rm -rf $E/stage
( cd $E/venv/lib/python3.13/site-packages && sha256sum --quiet -c $T/ref_site_packages.sha256 ) && echo "E1 site-packages: all $(wc -l < $T/ref_site_packages.sha256) .py/.so/.pth files identical to the 4090"
( cd $E && sha256sum --quiet -c $T/ref_xvla.sha256 ) && echo "E1 c8old/xvla: all $(wc -l < $T/ref_xvla.sha256) .py identical to the 4090"
$E/venv/bin/python -c "import sys, torch, lerobot, transformers; print('E2', sys.version.split()[0], sys.prefix, torch.__version__, transformers.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_capability() if torch.cuda.is_available() else None)"
