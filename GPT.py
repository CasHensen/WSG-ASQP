from langchain.callbacks import get_openai_callback
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

import pandas as pd
import numpy as np 
from numpy import exp
import os
from tqdm import tqdm
import logging
from langchain.evaluation import load_evaluator

import data_utils

class gpt:
    """
    This class carries out the GPT approach for both the labeling and the inference 

    The GPT approach uses one of two techniques either just the few shot approach or the evaluator technique, which adds an evaluator to the few shot gpt chain. 

    attributes: 
    - category_seeds: a dictionary containing the aspect categories and its seeds words, but only the aspect categories are used 
    - sentiment_seeds: a dictionary containing the sentiment polarities and its seeds words, but only the sentiment polarities are used 
    - gpt: the file containing the environment variables for the GPT model
    - args: the arguments containing the technique to use, but also the thresholds and other parameters.  
    """
    def __init__(self, args, category_seeds, sentiment_seeds):
        """
        The intialization function to assign variables to the class attributes.

        input: 
        - args: the arguments containing which technique to use, but also the thresholds and other parameters. 
        - category_seeds: the seed words for the aspect categories
        - sentiment_seeds: the seed words for the sentiment polarities
        """ 
        self.category_seeds = category_seeds
        self.sentiment_seeds = sentiment_seeds
        self.gpt = args.gpt
        self.args = args


    def __call__(self, dataset):  
        """
        The call function of the class executes the GPT approach

        input: 
        - dataset: the dataset on which the inference or labeling needs to be done
        """
        category_chain, evaluator = self.create_chain()
        labeled = []
        for sentence in tqdm(dataset):
            quads = self.get_quads(sentence['sentence'], category_chain, evaluator)
            label = {'sentence': sentence['sentence'], 'quads': quads, 'label': sentence['label']}
            labeled.append(label)
        return labeled


    def create_chain(self):
        """
        Creates the gpt chain through which the datasets will go 

        output: 
        - category_chain: the flow through the GPT model containing the prompt, the actual model, and the parser to make sure the output is in the correct format. 
        - evaluator: The evaluator module added on top of the category chain to obtain the evaluator technique
        """
        load_dotenv(self.gpt, override=True)
        parser = JsonOutputParser(pydantic_object=data_utils.Quadruplets)
        model = AzureChatOpenAI(
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_LABEL"),
                model_name=os.getenv("AZURE_OPENAI_MODEL_NAME_LABEL"),
                azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT_LABEL"),
                openai_api_key=os.getenv("AZURE_OPENAI_API_KEY_LABEL"),
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION_LABEL"),
                temperature=0,
                max_retries=6,
            )
        self.prompt = self.load_prompt(parser)
        
        category_chain = self.prompt | model | parser

        hh_criteria = {
            "certainty": """
                score 10: means you are extremely certain about all the elements in the quadruplet and all elements are necessary to mention, 
                score 8: means you are uncertain about one element of either the aspect term or the opinion term but you are still certain about the aspect category and sentiment polarity, 
                score 6: means you are uncertain about both the aspect term and opinion term, but you are still certain about the aspect category and sentiment polarity, 
                score 4: means you are uncertain about one element of either the aspect category or sentiment polarity, 
                score 2: when you are uncertain about both the aspect category and sentiment polarity. 
                score 0: You are uncertain about all the elements in the quadruplet
            """
        }

        evaluator = load_evaluator("score_string", criteria=hh_criteria, llm=model,)
        return category_chain, evaluator

    def get_quads(self, sentence, category_chain, evaluator):
        """
        The function returns the quadruples from the sentence as extracted by the GPT approach using either the few shot technique or the evaluator technique 

        input: 
        - sentence: the sentence from which the quadruplets needs to be extracted 
        - category_chain: the gpt chain to extract the quadruplets from the sentence using the few shot approach
        - evaluator: the evaluator module which is added on top of the few shot technique to get the evaluator technique 

        output: 
        - quads: the quadruplets extracted from the sentence, in the form of a dictionary containing the inputs: aspect, category, opinion, and sentiment 
        """
        quads = []
        try:
            with get_openai_callback() as cb:
                result = category_chain.invoke(
                    {
                        "category_seedwords": list(self.category_seeds.keys()),
                        "sentiment_seedwords": list(self.sentiment_seeds.keys()),
                        "sentence": sentence,
                    }
                )
                for label in result["quads"]:
                    if self.args.labeling == 'no_evaluator':
                        if label["aspect"]:
                            a = label["aspect"]
                            c = label["category"]
                            o = label["opinion"]
                            s = label["sentiment"]
                            quad = [a, c, s, o]
                        else:
                            quad = []
                    else:
                        eval_result = evaluator.evaluate_strings(prediction = label, input=self.prompt)
                        if eval_result['score'] > 7:
                            a = label["aspect"]
                            c = label["category"]
                            o = label["opinion"]
                            s = label["sentiment"]
                            quad = [a, c, s, o]
                        else: 
                            quad = []
                            
                    quads.append(quad)

        except Exception as e:
                logging.warning(e)
                logging.warning(f"the sentence with a mistake {sentence}")
                quads.append([])
        
        return quads


    def load_prompt(self, parser):
        """
        Load the prompt for the correct dataset 

        input: 
        parser: the parser to make sure the output is in the correct format

        ouput: 
        -prompt: the prompt format for the dataset evaluated 
        """
        if self.args.dataset == 'odido':
            prompt_cate = """
                You will get a summary divided per sentence. 
                Each summary is extrated from a phone call between a customer and an agent. 
                We want you to label the sentiment quadruplets in each sentence. 
                A quadruplets consists of an 'aspect', 'category', 'opinion', and 'sentiment'. 
                The 'aspect' and 'opinion' are words explicitly mentioned in the sentence, so need to be present in '''{sentence}'''.
                The 'aspect' are words in the sentence which inidcates which category is discussed. The 'opinion' are words in the sentence which indicates the sentiment towards the aspect and the category.  
                The 'category' and 'sentiment' assigned to quadruplets need to be present in the predefined lists. 
                For 'category' this list is: '''{category_seedwords}''' and for the 'sentiment' the list is: '''{sentiment_seedwords}'''
                The 'category' is are predefined clusters which indentify the elements of service provided. The 'sentiment' is a predefined list of polarities such as positive and negative. 
                Each sentence can contain zero, one or multiple quadruplets. 
                Now you will be shown 10 examples on how to assign labels to sentences. where first the sentence is given after "sentence: " and then the expected ouput is given after "output: "
                These examples can help guide you in assigning labels to the new sentence in '''{sentence}'''


                The sentence you will need to label are: 
                    '''{sentence}'''
                
               
                The format instructions are given in:
                 '''{format_instructions}'''

            """
        elif self.args.dataset == 'rest15' or self.args.dataset == 'rest16':
            prompt_cate = """
                You will get reviews divided per sentence. 
                Each reviw is on a restaurant and its services. 
                We want you to label the sentiment quadruplets in each sentence. 
                A quadruplets consists of an 'aspect', 'category', 'opinion', and 'sentiment'. 
                The 'aspect' and 'opinion' are words explicitly mentioned in the sentence, so need to be present in '''{sentence}'''.
                The 'aspect' are words in the sentence which inidcates which category is discussed. The 'opinion' are words in the sentence which indicates the sentiment towards the aspect and the category.  
                The 'category' and 'sentiment' assigned to quadruplets need to be present in the predefined lists. 
                For 'category' this list is: '''{category_seedwords}''' and for the 'sentiment' the list is: '''{sentiment_seedwords}'''
                The 'category' is are predefined clusters which indentify the elements of service provided. The 'sentiment' is a predefined list of polarities such as positive and negative. 
                Each sentence can contain zero, one or multiple quadruplets. 
                Now you will be shown 10 examples on how to assign labels to sentences. where first the sentence is given after "sentence: " and then the expected ouput is given after "output: "
                These examples can help guide you in assigning labels to the new sentence in '''{sentence}'''
                
                1. 
                sentence: i asked for a menu and the same waitress looked at my like I was insane
                output: 
                {{'quads':[
                    {{'aspect': 'waitress', 'category': 'service', 'opinion': 'insane', 'sentiment': 'negative'}}
                    ]
                }}
                
                2. 
                sentence: one of my favorite places in Brooklyn
                output:
                {{'quads':[
                    {{'aspect': 'NULL', 'category': 'restaurant', 'opinion': 'favorite', 'sentiment': 'positive'}}
                    ]
                }}
                   
                3. 
                sentence: service however was excellent and i liked the setting atmosphere a lot 
                output: 
                {{'quads':[
                    {{'aspect': 'service', 'category': 'service', 'opinion': 'excellent', 'sentiment': 'positive'}}, 
                    {{'aspect': 'setting', 'category': 'ambience', 'opinion': 'liked', 'sentiment': 'positive'}}
                    ]
                }}
                
                4. 
                sentence: the bagels always warm soft on the inside crispy on the outside and enormous in size 
                output: 
                {{'quads':[
                    {{'aspect': 'bagels', 'category': 'food', 'opinion': 'warm', 'sentiment': 'positive'}}, 
                    {{'aspect': 'bagels', 'category': 'food', 'opinion': 'soft', 'sentiment': 'positive'}}, 
                    {{'aspect': 'bagels', 'category': 'food', 'opinion': 'crispy', 'sentiment': 'positive'}}, 
                    {{'aspect': 'bagels', 'category': 'food', 'opinion': 'enormous', 'sentiment': 'positive'}}
                    ]
                }}
                
                5. 
                sentence: we were seated outside and the waiter spilled red wine and hot tea on myself and my date 
                output: 
                {{'quads':[
                    {{'aspect': 'waiter', 'category': 'service', 'opinion': 'spilled', 'sentiment': 'negative'}}
                    ]
                }}
                
                6. 
                sentence: good music great food speedy service affordable prices
                output: 
                {{'quads':[
                    {{'aspect': 'music', 'category': 'ambience', 'opinion': 'good', 'sentiment': 'positive'}}, 
                    {{'aspect': 'food', 'category': 'food', 'opinion': 'great', 'sentiment': 'positive'}}, 
                    {{'aspect': 'service', 'category': 'service', 'opinion': 'speedy', 'sentiment': 'positive'}}, 
                    {{'aspect': 'prices', 'category': 'price', 'opinion': 'afforable', 'sentiment': 'positive'}}
                    ]
                }}

                7. 
                sentence: the hostess and the waitress were incredibly rude and did everything they could to rush us out
                output: 
                {{'quads':[
                    {{'aspect': 'hostess', 'category': 'service', 'opinion': 'rude', 'sentiment': 'negative'}}, 
                    {{'aspect': 'waitress', 'category': 'service', 'opinion': 'rude', 'sentiment': 'negative'}}
                    ]
                }}

                8. 
                sentence: the restaurant is a bit noisy but that is something that can be overlooked once you sit down and enjoy a great meal
                output: 
                {{'quads':[
                    {{'aspect': 'meal', 'category': 'food', 'opinion': 'enjoy', 'sentiment': 'positive'}}, 
                    {{'aspect': 'meal', 'category': 'food', 'opinion': 'great', 'sentiment': 'positive'}},
                    {{'aspect': 'restaurant', 'category': 'ambience', 'opinion': 'noisy', 'sentiment': 'negative'}}
                    ]
                }}
                
                9. 
                sentence: the food however is what one might expect 
                output: 
                {{'quads':[
                    {{'aspect': 'food', 'category': 'food', 'opinion': 'expect', 'sentiment': 'negative'}}
                    ]
                }}

                10. 
                sentence: good spreads great beverage selections and bagels really tasty
                output: 
                {{'quads': [
                    {{'aspect': 'spreads', 'category': 'food', 'opinion': 'good', 'sentiment': 'positive'}}, 
                    {{'aspect': 'beverage', 'category': 'drinks', 'opinion': 'great', 'sentiment': 'positive'}},
                    {{'aspect': 'bagels', 'category': 'food', 'opinion': 'tasty', 'sentiment': 'positive'}}
                    ]   
                }}
                
                
                The sentence you will need to label is: 
                    '''{sentence}'''


                The format instructions are given in:
                 '''{format_instructions}'''
                
               
            """
        

        prompt = PromptTemplate(
            input_variables=["category_seedwords", "sentiment_seedwords", "sentence"],
            template=prompt_cate,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        
        return prompt
