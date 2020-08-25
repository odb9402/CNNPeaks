gcc -g -O2 -Wall -pthread -c readbam.c -lz -lpthread -lm
ar rcs libreadbam.a readbam.o
python3 setup.py build_ext --inplace
rm -r build
