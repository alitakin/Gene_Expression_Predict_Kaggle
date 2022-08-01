# Gene_Expression_Predict_Kaggle

Histone modifications are playing an important role in affecting gene regulation. Nowadays, predicting gene expression from histone modification signals is a widely studied research topic.

The dataset of this competition is on "E047" (Primary T CD8+ naive cells from peripheral blood) celltype from Roadmap Epigenomics Mapping Consortium (REMC) database. For each gene, it has 100 bins with five core histone modification marks [1]. (We divide the 10,000 basepair(bp) DNA region (+/-5000bp) around the transcription start site (TSS) of each gene into bins of length 100 bp [2], and then count the reads of 100 bp in each bin. Finally, the signal of each gene has a shape of 100x5.)

The goal of this competition is to develop algorithms for accurate predicting gene expression level. High gene expression level corresponds to target label = 1, and low gene expression corresponds to target label = 0.

link of competition in kaggle website: [gene expression prediction competition] (https://inclass.kaggle.com/c/gene-expression-prediction)
read more about the data: [DeepChrome](https://arxiv.org/abs/1607.02078)
## To Do 
* Trying other algorithms.
* Ploting learnig curve.
* Finding a way for data augmantation.
* Switching to Deep Learning algorithms. 

## Built With
* [Python 2.7 & 3.5] (https://www.python.org/doc/)
* [scikit-learn 0.18](http://scikit-learn.org/stable/documentation.html) - The machine learning framework used


## Authors

* **Pia Niemala**  - pia.s.niemala@gmail.com
* **Zeinab R.Yousefi**  - z.rezaei@gmail.com
* **Azarkhsh Hamedi**  - azarakhsh.h@gmail.com
* **Saboktakin Hayati**  - saboktakinhayati@gmail.com


See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
* Tampere university of technology
* [Department of Signal Processing](http://www.tut.fi/en/about-tut/departments/signal-processing/), [SGN-41007 course](http://www.cs.tut.fi/courses/SGN-41007/)  and the lectrure [Heikki Huttunen](http://www.cs.tut.fi/~hehu/), which made us to participate in this competition :). 
* Hat tip to anyone who's code was used
* Inspiration
* etc