#!/bin/bash -x

#for TIFM
Sz=-1
Num_sites=10
Dmax=22

cd runForInput || exit

for H in $(seq 0.0 0.05 1.8)
do
  mkdir -p H_$H
  cd H_$H || exit
  cp ../../input_template.inp ./input.inp

  sed -i 's/fieldfield/'-$H'/g' input.inp
  sed -i 's/szszsz/'$Sz'/g' input.inp
  sed -i 's/numofsites/'$Num_sites'/g' input.inp
  sed -i 's/maxnumofstates/'$Dmax'/g' input.inp

  # shellcheck disable=SC2046
  python ../../main.py $(pwd)/input.inp &> runForInput.cout
  echo "H_$H finished"
  cd ..


done
cd ../entanglement
python entspec.py











