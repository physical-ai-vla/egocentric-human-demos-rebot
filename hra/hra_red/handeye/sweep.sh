#!/bin/bash
# x y z pitch roll
i=0
while read x y z p r; do
  i=$((i+1)); n=$(printf "hc%02d" $i)
  res=$(curl -s -m 180 -X POST "localhost:8056/hra_start_pose?pitch=$p&roll=$r&x_mm=$x&y_mm=$y&z_mm=$z&steps=6" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('error') or ('ok tcp %s pitch %s roll %s'%(d['tcp_mm_now'],d['pitch_now_deg'],d['roll_now_deg'])))")
  sleep 1.5; echo "$n [$x $y $z p$p r$r] $res :: $(./snap.sh $n)"
  case "$res" in *rror*|*failed*) echo STOP; break;; esac
done <<'P'
300 -180 200 55 0
300 -180 200 55 15
300 -180 200 55 -15
270 -180 200 50 0
330 -180 210 62 0
300 -150 200 55 10
300 -210 200 55 -10
280 -160 170 50 20
320 -200 230 60 -20
300 -180 240 60 0
300 -180 170 48 0
270 -210 220 52 -15
330 -150 190 60 15
290 -170 210 65 0
310 -190 190 45 10
300 -180 200 55 0
P
