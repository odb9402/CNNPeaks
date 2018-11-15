#!/bin/bash

while [ -n "$1" ]
do
    case $1 in

    -t)
        echo "Threshold is $2"
        inputT=$2
        shift 2
        ;;

    -i)
        echo "Input is $2"
        input=$2
        shift 2
        ;;
    -o)
        echo "Output is $2"
        output=$2
        shift 2
        ;;
esac
done


if [ $? != 0 ]; then usage; fi

# Remove peaks that have interval smaller than threshold value.
awk -v threshold=$inputT '{ if(($3-$2)>threshold) {print $0} }' $input > $output
