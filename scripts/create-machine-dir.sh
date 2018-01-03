#!/bin/sh

NUMPROC=2

if [[ -d machinefiles ]]
then
    rm -ir machinefiles
fi

mkdir machinefiles
cd machinefiles

#for i in {2..254}
#do
#    echo "137.44.6.${i}" >> ./machine.file.FULL
#done
#machinelist="./machine.file.FULL"

# Above is more thorough, but this is easier and quicker 
machinelist="/home/XXX/machine.list"

# SSH test each machine
while read -u10 ipaddress; 
do
	ssh -o "StrictHostKeyChecking no" -o BatchMode=yes -o ConnectTimeout=1 $ipaddress "hostname -i;"
	
done 10< $machinelist > ./machine.file.NEW

# Create files
for i in {1..6}
do
    cp machine.file.NEW ${NUMPROC}.mach.list
    sed -i "$((NUMPROC+1)),\${s/^/#/}" ${NUMPROC}.mach.list
    let NUMPROC=$NUMPROC*2
done

