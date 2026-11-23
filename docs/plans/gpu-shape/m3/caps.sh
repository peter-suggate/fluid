#!/bin/zsh
# Thread caps (Metal, size attribute stripped) of the M3 shape variants (dam's lowered modules) and the bases (M2's).
D=${0:A:h}; cd $D/caps
cp $D/dam/wgsl/uTransfer.m3.{first,app2,app4,app8,first2,first4,first8}.85.wgsl $D/dam/wgsl/uAdvect.m3.{first,app2,app4,app8,first2,first4,first8}.83.wgsl .
cp $D/../m2/caps-dam/uTransfer-base.wgsl $D/../m2/caps-dam/uAdvect-base.wgsl $D/../m2/caps-dam/uExtend-base.wgsl .
: > caps.txt; for f in *.wgsl; do echo -n "$f " >> caps.txt; zsh ~/code/me/fluid-scratch/brick-tune/regcap.sh $f >> caps.txt 2>&1; done
