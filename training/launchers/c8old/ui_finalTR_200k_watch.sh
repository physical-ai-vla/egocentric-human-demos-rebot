#!/bin/bash
# launch the B1-old UI for final TR 200k on :8016 once the SSD copy is sha-verified (.completed)
D=/Volumes/PortableSSD/rebot_ckpts_archive/5090_c8old_c8oldv2_finalTR_300k
until [ -f $D/200000.completed ]; do sleep 60; done
cd ~/holobrain-mac-model && XVLA_CKPT=$D/200000/pretrained_model E280_CAMS=middle,left,right E280_PROMPT_STYLE=robot E280_CLOSED_L=58 E280_CLOSED_R=68 HB_UI_PORT=8016 \
  nohup ~/groot-infer-env/bin/python ~/holobrain-mac-model/mac_infer_ui_e280.py > ~/c8oldTR200k_ui_8016.log 2>&1 &
echo "launched 200k UI :8016 at $(date '+%F %T')"
