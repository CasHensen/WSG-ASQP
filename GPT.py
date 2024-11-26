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
    def __init__(self, args, category_seeds, sentiment_seeds):
        self.category_seeds = category_seeds
        self.sentiment_seeds = sentiment_seeds
        senttag2opinion, self.sentword2opinion, opinion2word = data_utils.get_sentiment(args)
        self.gpt = args.gpt
        self.args = args


    def __call__(self, dataset):  
        category_chain, evaluator = self.create_chain()
        labeled = []
        for sentence in tqdm(dataset):
            quads = self.get_quads(sentence['sentence'], category_chain, evaluator)
            label = {'sentence': sentence['sentence'], 'quads': quads, 'label': sentence['label']}
            labeled.append(label)
        return labeled


    def create_chain(self):
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

    def create_target(self, quads):
        all_quad_sentences = []
        for quad in quads:
            a, c, s, o = quad

            try:    
                s_change = self.sentword2opinion[s]  # 'POS' -> 'good'
            except: 
                s_change = s 

            if a == 'NULL':  # for implicit aspect term
                if self.args.dataset == 'rest15' or self.args.dataset == "rest16":
                    a = 'it'
                elif self.args.dataset == "odido":
                    a = 'het'
            if a:
                if self.args.dataset == 'rest15' or self.args.dataset == "rest16":  
                    one_quad_sentence = f"{c} is {s_change} because {a} is {o}"
                elif self.args.dataset == "odido":
                    one_quad_sentence = f"{c} is {s_change} omdat {a} is {o}"
            else: 
                one_quad_sentence = ""

            all_quad_sentences.append(one_quad_sentence)
            
        target = ' [SSEP] '.join(all_quad_sentences)
        return target


    def load_prompt(self, parser):
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
