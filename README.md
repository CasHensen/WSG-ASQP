# WG-ASQP

This repository contains the code for the Thesis From Words To Insights: A Weakly Generative Aspect Sentiment Quad Prediction Method by Cas Hensen. 

After installing the required packages from the requirement.txt, the code can be run from the main.py code file. 
In this file the parameters are initialized and they can be adjusted in the first function. 
In the data folder the SemEval data used in the thesis is stored. 

In the main.py the rest of the code files are called upon in the order for the WG-ASQP, however the code files can also be called upon individually.  
The Labeler.py contains the class, which labels the data. From this class either the class in BERT_approach.py or the class in GPT.py is called to automatically label the unlabeled dataset. 
The class in the BERT_approach.py, then, calls the steps for the BERT labeling approach. These steps consist of the creation of vocabularies with Vocabulary_generator.py, the computation of the overlap score with Score_computer.py and the assignment of labels with Assign_labels.py. 
The GPT.py contains all the functions for the automatic labeling itself, so no other coding files need to be called upon. 

If the automatically labeled data is obtained, it is used to post-train a generative method using the Generative_model.py. The generative model makes the distinction whether you use the Paraphrase method by Zhang, Deng et al. (2021) or the GPT model is used. 
If the Paraphrase model is used, the class post-trains the pre-trained large language model at initialization of the class. Then you can choose for evaluation of the model on your dataset if you have manually labeled data or just inference on the dataset. 
If the GPT model is used, then no PLM is post-trained and you can imidiately use the evaluation or inference functions in the class. 

The helper functions are defined in three code files: data_utils.py, eval_utils.py, and save_utils.py. These files contain functions which are called upon throughout the different classes. 
