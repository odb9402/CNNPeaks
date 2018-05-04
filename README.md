CNN-peaks
===================



CNN-peaks is a **Convolution Neural Network(CNN)** based ChIP-Seq peak calling software. 

----------


Install
-------------

CNN-peaks has no extra install. You can use it just by making clone of this repository.

Labeled data
--------------- 
CNN-peaks uses labeled data which has its own format. All these approach that use labeled data for marking is from [1]. Examples of labeled data is as the following below. (It based on ASCII)

> 
	chr1:1,000,000-1,100,000 peaks K562
	chr1:1,100,000-1,200,000 peakStart K562
	chr1:1,250,000-1,300,000 peakEnd K562
	chr2:10,000,000-10,002,000 peaks


In line 1, **peaks**, this means that cell K562 at least has one peak in a region (chr1:1,000,000-1,100,000). In line 2, 3 ,**peakStart, peakEnd**,  it means that cell K562 just only one peak in the regions. In line 4, there is no peak in that regions about K562 or other cell because there is no matched cell line name at this raw. If you want to use this label data on other cells,  all these line 1-4 mean **noPeak** because there is no cell name in  the lines.  If you want to know specific rules or methods of this labeling work, please look [here.](https://academic.oup.com/bioinformatics/article/33/4/491/2608653/Optimizing-ChIP-seq-peak-detectors-using-visual)


Usage of CNN-peaks
---------------
**1. Training data to build a CNN-peaks model.**

Before you try to call peaks with your ChIP-Seq Data, CNN-peaks model should be trained by internal module in CNN-peaks. CNN-peaks gives **preprocessing** module for generating actual training data samples with labeled data and bam alignment files of them. You can use our labeled data sample or use your own labeled data to generate training data. When you use its preprocessing module, the input of the module is a single directory which includes labeled data and bam files. For example, a directory which has name "./TestPreProcess" includes these files:


> **./TestPreProcess:**

> - H3K36me3_None.txt
> - H3K36me3_K562.bam
> - H3K36me3_A549.bam

Each file must follow a format of filename that is **Target_CellName.bam**. The case of labeled data, it must be like this :**Target_AnyString.txt**. The labeled data of this example H3K36me3_None.txt has data which includes both of K562 and A549.
