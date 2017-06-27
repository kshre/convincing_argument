"""
This file will use pretrained word embeddings to prepare representations
of the argument and the wikipedia articles, then do a sum over the dot
product of the representation of the argument with each wikipedia article
and finally use the sum as the representation of argument combined with
knowledge from wikipedia. This final combined representations of two
articles will be used to do classification between the two for which one
is more convincing using logistic regression.
"""
