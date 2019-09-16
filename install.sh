sudo apt-get install -y --no-install-recommends bamtools \
    libbz2-dev \
    liblzma-dev
    

### INSTALL HTSLIB 
wget https://github.com/samtools/htslib/releases/download/1.9/htslib-1.9.tar.bz2
tar -xvf htslib-1.9.tar.bz2
cd htslib-1.9.tar.bz2
sudo ./configure
sudo make
sudo make install
cd ..

pip install numpy scipy sklearn tensorflow pandas progressbar2 pysam tensorflow-gpu

cd dependencies
./install_samtools.sh

cd ..
./build.sh
