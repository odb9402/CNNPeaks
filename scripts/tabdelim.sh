awk 'BEGIN {FS="[\t]"}; {print $0}' $1 > temp
bedtools sort -i temp > temp.sort
mv temp.sort $1
rm temp
