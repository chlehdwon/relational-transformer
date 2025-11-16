for task in driver-dnf driver-position driver-top3
do
    ./finetune.sh rel-f1 $task
done

for task in user-clicks user-visits ad-ctr
do 
    ./finetune.sh rel-avito $task
done
