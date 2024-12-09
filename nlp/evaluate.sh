#!/bin/bash

# Get the operating system name
os_name=$(uname -s)

# Check if the OS is Windows (Cygwin or WSL)
if [ "$os_name" == MSYS_NT* ]; then
  ROOT_DIR="C:/Users/O772985/OneDrive - JPMorgan Chase/MSDE/tmp/data_backup"
elif [ "$os_name" == "Linux" ]; then
  ROOT_DIR="/home/omniai-jupyter/nlp/pyanalysis/data_backup"
else
  ROOT_DIR="/Users/honghu/Doc/w266/Project/data_backup"
fi


find $ROOT_DIR -type f -name "evaluate.log" | xargs grep -H -e "Eval complete" | awk -F'/' '
{
    model_name = $(NF-3)"/"$(NF-2)"/"$(NF-1);
    split($(NF), metrics, " ; ");
    split(metrics[5], f1, ": ");
    split(f1[3], f1_val, ", ");
    print model_name, metrics[2], metrics[3], metrics[4], "overall_f1:", f1_val[1]
}' | awk '
{
    split($1, full_path, "/");
    print full_path[1]"/"full_path[2], $(NF), $0
}
' | sort -r | awk '{$1=""; $2=""; print $0}' OFS=" " |  sed 's/^..//'

