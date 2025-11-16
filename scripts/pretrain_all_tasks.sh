for task in driver-dnf driver-position driver-top3
do
    ./pretrain_contd.sh rel-f1 $task
done

for task in user-clicks user-visits ad-ctr
do 
    ./pretrain_contd.sh rel-avito $task
done
