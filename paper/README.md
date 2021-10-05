# WMDecompose: A Framework for Leveraging the Interpretable Properties of Word Mover’s Distance in Sociocultural Analysis

The WMDecompose Python library is introduced in a paper published in the Proceedings of [the 5th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature](https://www.aclweb.org/portal/content/5th-joint-sighum-workshop-computational-linguistics-cultural-heritage-social-sciences), a yearly workshop organized by the Association for Computational Linguistics (ACL). The paper aims 1) to explain the analytic motivation for creating the codebase, 2) to describe the pipeline in plain English, without direct reference to code, and 3) to illustrate its utility with a small vignette analysis. 

The [notebooks folder](https://github.com/maybemkl/wmdecompose/tree/master/paper/notebooks) contains Jupyter notebooks for example code and for replicating the results of the paper. Readers looking for a hands-on introduction to WMDecompose can find a fully coded example analysis using simple IMDB data in the IMDB_example.ipynb notebook. We recommend that new users begin here. The reddit_example_paper.ipynb replicates the published results, including all pre-processing and visualization, while Yelp_example.ipynb replicates Appendix B of the paper. The finetune_w2v_IMDB.ipynb and finetune_w2v_Yelp.ipynb notebooks containi code for finetuning the Google News word2vec model, which can be downloaded [here](https://github.com/mmihaltz/word2vec-GoogleNews-vectors).

The paper can be cited as follows:

Brunila, M., and LaViolette, J. (2021). WMDecompose: A Framework for Leveraging the Interpretable Properties of Word Mover’s Distance in Sociocultural Analysis. _Proceedings of the_5th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature_, START_PAGE–END_PAGE.

