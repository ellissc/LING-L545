Metrics    | Precision |    Recall |  F1 Score | AligndAcc
-----------+-----------+-----------+-----------+-----------
Tokens     |    100.00 |    100.00 |    100.00 |
Sentences  |    100.00 |    100.00 |    100.00 |
Words      |    100.00 |    100.00 |    100.00 |
UPOS       |    100.00 |    100.00 |    100.00 |    100.00
XPOS       |    100.00 |    100.00 |    100.00 |    100.00
Feats      |    100.00 |    100.00 |    100.00 |    100.00
AllTags    |    100.00 |    100.00 |    100.00 |    100.00
Lemmas     |    100.00 |    100.00 |    100.00 |    100.00
UAS        |     80.89 |     80.89 |     80.89 |     80.89
LAS        |     78.26 |     78.26 |     78.26 |     78.26

After looking at the generated trees, the trained udpipe appears to do a decent job, but there are still some errors. In trees #2,#7, and #11 there are different roots; tree #2 differs with an object being the root in one, while a clausal complement is the root in the other. In tree #11, a dislocated noun is the root, while a parataxis noun is the root in the other. Another error deals with nsubj organization and 在 (xcomp vs ccomp), as one tree attaches the nsubj to the overarching verb, while the other tree (correctly) attaches it to the 在 ccomp (tree #8). Another interesting error that occurs in tree #5 relates to how two verbs are connected; one tree has the second verb, parataxis, connect to the words after the comma, while the other tree has the second verb, ccomp, connects to all but two words, even those before the comma.
