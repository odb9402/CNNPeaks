tar -xvf samtools-1.8.tar.bz2

cd samtools-1.8

cd htslib-1.8

./configure --prefix=/usr/include/bin
make
make install
export PATH=/usr/include/bin:$PATH

cd ..
make
make install
./configure --prefix=/usr/bin
export PATH=/usr/bin:$PATH

