Tools Used
===========

Programing Language: Pyhton (Version 3.6.2)
-------------------
Package for stop words: Python's stop-words package (https://pypi.python.org/pypi/stop-words)
-----------------------


How To Invoke
=============

- Open command prompt
- Install the stop-words package by using the following command,
	=> pip install stop-words

- Invoke the Python file 'MNB.py' by passing the following command
	=> Python MNB.py 20news-bydate-train 20news-bydate-train

- The two arguments passed,
	=> Path to Training data
	=> Path to Test Data 


Results & Analysis
==================

Following outputs were observed when running with 4 different subsets(5 out of 20) of the 20 newsgroups datasets.

													 Datasets 																	 Accuracy
		 ==================================================================================================================		==========
1- 		 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian' 		 								  89.78 %
2- 		 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc', 'soc.religion.christian'	  79.58 %
3-		 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'								  89.70 %
4- 		 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x' 		  91.61 %


Average accuracy = 87.66 %