#!/bin/bash -x

#for TIFM
Sz=-1
Num_sites=8
Dmax=22

cd runForInput || exit

for H in $(seq 0.9 0.05 1.1)
do
  mkdir -p H_$H
  cd H_$H || exit
  cp ../../input_template.inp ./input.inp

  sed -i 's/fieldfield/'$H'/g' input.inp
  sed -i 's/szszsz/'$Sz'/g' input.inp
  sed -i 's/numofsites/'$Num_sites'/g' input.inp
  sed -i 's/maxnumofstates/'$Dmax'/g' input.inp

  # shellcheck disable=SC2046
  python ../../main.py $(pwd)/input.inp &> runForInput.cout
  cd ..


done
cd ..











