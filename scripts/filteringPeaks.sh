#!/bin/bash

function description
{
        echo -e "\t<USAGE of filteringPeaks>"
        echo -e "\t:It will remove peaks by options.\n"
        echo -e "\t[filteringShortPeaks -o<options> -t <threshold> -i <input> > <output>]\n"
        echo -e "\t:[-o] interval  : filtering peaks by those interval (3``th col - 2``th col)"
        echo -e "\t      sigValue  : filtering peaks by sigmoid activation of peak regions (5``th col)"
        echo -e "\t      pValue    : filtering peaks by those pvalue is bigger than threshold. (6``th col)"
        echo -e "\t      score     : filtering peaks by score of peak regions (8``th col)"
        echo -e "\t:[-t] threshold value for [-o]\n"
        echo -e "\t:[-i] input_file name\n"
}

while [ -n "$1" ]
do
    case $1 in

    -h)
        description
        exit 1
        ;;

    -o)
        thsOpt=$2
        shift 2
        ;;
    -t)
        inputT=$2
        shift 2
        ;;

    -i)
        input=$2
        shift 2
        ;;
    *)
        echo "you used wrong options $1"
        description
        exit 1
        ;;
esac
done


if [ $? != 0 ]; then usage; fi

# Remove peaks that have interval smaller than threshold value.
case $thsOpt in
    interval)
        awk -v threshold=$inputT 'BEGIN {FS="[\t]"}; {if(($3-$2)>threshold) {print $0}}' $input
        ;;
    pvalue)
        awk -v threshold=$inputT 'BEGIN {FS="[\t]"}; {if(($6)<threshold) {print $0}}' $input
        ;;
    sigValue)
        awk -v threshold=$inputT 'BEGIN {FS="[\t]"}; {if(($5)>threshold) {print $0}}' $input
        ;;
    score)
        awk -v threshold=$inputT 'BEGIN {FS="[\t]"}; {if(($8)>threshold) {print $0}}' $input
        ;;
    *)
        echo "Please select option among {interval, pvalue, sigValue, avgDepth, maxDepth}"
        description
        exit 1
        ;;
esac
