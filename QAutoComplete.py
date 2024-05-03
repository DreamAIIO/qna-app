import json
import os
import pickle
# import re
from collections import OrderedDict

from rapidfuzz import fuzz, process
# from rapidfuzz.distance import DamerauLevenshtein as dlvs
# from rapidfuzz.distance import Hamming as ham
from rapidfuzz.distance import OSA as osa
from rapidfuzz.distance import Levenshtein as lvs


def read_json(filename):
    with open(filename) as f:
        data = json.load(f)
    return data

def read_pickle(filename):
    with open(filename,'rb') as f:
        data = pickle.load(f)
    return data

def read_text(filename):
    with open(filename,'r') as f:
        data = [l.strip() for l in f.readlines()] # read file line-by-line and return list of strings
    return data
    
class AutoComplete:

    def __init__(self,use_clm:bool=False,model_path:str="distilgpt2-QuesAutoComplete",checkpoint='distilgpt2',sents_data_path:str='sents_data.pkl'):
        self.use_clm = use_clm
        if self.use_clm and os.path.exists(model_path):
            from transformers import pipeline

            #checkpoint = model_path.split('/')[-1] # get the model name
            #checkpoint  = '-'.join(checkpoint.split('-')[:-1]) # get the checkpoint name
            self.generator = pipeline('text-generation', model=model_path,tokenizer=checkpoint)
            print("Text Generation Model Loaded!")
        else:
            # self.nlp = spacy.load('en_core_web_lg')
            if os.path.exists(sents_data_path):
                if sents_data_path.endswith('.pkl'):
                    self.sents_data = read_pickle(sents_data_path)
                elif sents_data_path.endswith('.txt'):
                    self.sents_data = read_text(sents_data_path)
                self.sents_data_path = sents_data_path
            else:
                self.sents_data_path = ''
                self.sents_data = []
                print(f"{sents_data_path} not found! Using default data!")
            self.sents_data += [
                "list all items",
                "list all items with price greater than $100",
                "how many items have price greater than $100",
                "list all items with po cost lower than $10",
                "list all items with po cost per unit greater than $20",
                "list all items with store cost higher than $50",
                "list all items that have freight higher than $1",
                "which items have everyday margin greater than 50",
                "list all items that have type R",
                "list items that have size 15 OZ",
                "which items belong to the Bakery category?",
                "how many items belong to the Bakery category?",
                "which items have category Deli?",
                "which item has vendor item number 1106L?",
                "which category does salsa original belong to?",
                "list all items from the Deli category with price greater than $100.",
                "list all items from the Dairy category with po cost lower than $10.",
                "list all items from the Bakery category with store cost greater than $10.",
                "list all items from the Grocery category with everyday margin higher than 35.",
                "what is the upc of cheesecake cone",
                "what is store cost of items sold by rudolph foods",
                "what is the vendor item number of pepperoni sandwich",
                "what is the size and po cost per unit of salad beet?",
                "what is the product code of items with po cost lower than 5 and freight cost greater than 0?",
                "what is the address and zip code of happy cow?",
                "what is the zipcode of customers that have city Harper.",
                "how many items are sold by rudolph foods",
                "which vendor sells beef sticks?",
                "list items sold by medina foods that have price greater than $10",
                "how many items are sold by medina foods that have price greater than $10",
                "list items sold by perdue farms that have po cost per unit greater than $5 and freight cost greater than $1",
                "how many items were sold by perdue farms that have po cost per unit greater than $5 and freight cost greater than $1",
                "list all items sold by distinctive foods with store cost lower than $40 that belong to the Grocery category",
                "list all items from the Meat category sold by American Food Group",
                "what is the total price of items sold by vendor 4879?",
                "what is the mean po cost per unit of items sold by vendor 3480",
                "list all items bought by Gordon Food Service",
                "list all customers that bought liver calf",
                "list items bought by Nabby's Market that have price greater than $10",
                "list all customers that purchased items with po cost per unit greater than $5 and store cost lower than $5",
                "what is the max store cost of items purchased by customer 26005000?",
                "what is the address of customers that bought item calzone?",
                "what is the state and city of customers that bought meatloaf with ketchup glaze",
                "list all vendors that sold to Grab N Dash",
                "list all vendors that The Gin bought from",
                "list all customers that vendor Hoff's Bakery sold to",
                "list all items that vendor perdue sold to customer harvest market",
                "list all items that family foods purchased from smithfield",
                "list items sold by Farm Ridge and bought by Winey Cow",
                "list customers that purchased items from the Grocery category sold by Distinctive Foods.",
                "What is the average store cost of items that customer 81297000 bought from vendor 4065 that belong to the Seafood category and have price lower than $10.",
                "list items bought by Fresh Take and sold by Hoff's Bakery that have po cost greater than $50",
                "list all items with sales handled by sales rep Davin",
                "list the sales representatives that handled the sales for Salami Hard",
                "list all items that were sold in the sales region Georgia",
                "list the sales regions where Olives Stuffed Blue was sold",
                "list all items with sales handled by sales rep David in the sales region OH South",
                "list all vendors from records with record weight greater than 1500 and record cases greater than 50",
                "what is the average weight bought by customer 95452006",
                "list all customers from records with record weight greater than 1500 and record cases greater than 50",
                "list all customers who had sales handled by sales rep Ray",
                "list all items that were sold by Cooper Farms in May 2022",
                "list all items that were sold by vendor 5571 with sales handled by sales rep Day",
                "list the sales representatives that handled the sales for item Salsa bought by customer Family Foods",
                "list all customers that bought item Salsa with the sales handled by sales rep Annie in the sales region IL East",
                "list all customers that bought from Perdue Farms in March 2021",
                "list all records which record the customer Winey Cow and vendor Depalo Foods",
                "what is the average record weight that customer 95452006 bought from vendor 3480",
                "what is the record weight of all records that record the customer 95452006 and vendor 3480",
                "list all customers from records with record weight lower than 2000 and record cases greater than 100.",
                "list all customers that bought from vendor Smithfield with sales handled by sales rep Davin",
                "list all customers that bought from vendor Willy's Salsa in the sales region OH South",
                "list all vendors that sold to customer Winey Cow in the sales region East",
                "how many vendors sold to customer Winey Cow in the sales region East",
                "list all sales representatives that handled sales in sales region ky/ty/in in 2022"
            ]
            self.sents_data = set(self.sents_data)
    
    def generate_suggestions(self,q:str,top_k:int=5,**kwargs) -> dict:
        '''
        Given an incomplete query, returns the best 'top_k' suggestions
        
        Parameters:-
            q: query/question to generate suggestions for
            top_k: Maximum number of 'best' suggestions to return
            score_cutoff: Only return suggestions that have a higher similarity score than 'score_cutoff'
            
        Returns:- A dictionary with the following key(s):
                i) suggestions: List of strings/suggestions of the most similar queries
                ii) heading: An empty string
                iii) rows: An empty list
        '''
        if q.endswith('.') or q.endswith('?'):
            return {'suggestions':[q],'heading':"",'rows':[]}
        if len(q.split(' ')) < 2:
            return {'suggestions':[],'heading':"No suggestions available!",'rows':[]}
        if self.use_clm:
            from transformers import set_seed
            set_seed(5783)
            suggs = [s['generated_text'] for s in self.generator(q.strip(' \n\t'), max_length=50,
                                            do_sample=True,
                                            top_k=5,
                                            #no_repeat_ngram_size=1,
                                            num_return_sequences=10)]
            suggs = list(OrderedDict({s:'' for s in suggs}).keys())
            return {'suggestions':suggs,'heading':"",'rows':[]}
        else:
            return self.__generate_suggestions_lex(q,top_k,kwargs.get('score_cutoff',len(q)*0.5))
    
    def __generate_suggestions_lex(self,q:str,top_k:int=25,score_cutoff:int=10) -> dict:
        '''
        Use sentences from training data and measures the lexical similarity between the sentences and
        input question to select the best 'top_k' suggestions that have a similarity score higher than score_cutoff
        
        Parameters:-
            q: query/question to generate suggestions for
            top_k: Maximum number of 'best' suggestions to return
            score_cutoff: Only return suggestions that have a higher similarity score than 'score_cutoff'
            
        Returns:- A dictionary with the following key(s):
                i) suggestions: List of strings/suggestions of the most similar queries
        '''
        # A custom scoring function that combines levenshtein similarity, OSA similarity and token_set_ratio
        def scorer_fn(s1,s2,processor=lambda a: a.lower(),score_cutoff=5):
            
            if s1 == '' or s2 == '':
                return 0
            # rep1 = self.nlp(s1)
            # rep2 = self.nlp(s2)
            
            #s1 = s1[:lq]
            #s2 = s2[:lq]
            if processor is not None:
                s1 = processor(s1)
                s2 = processor(s2)
            
            if len(s1) < len(s2):
                q_nwords = len(s1.split(' '))
                lq = len(' '.join(s2.split(' ')[:q_nwords+1]))
            else:
                q_nwords = len(s2.split(' '))
                lq = len(' '.join(s1.split(' ')[:q_nwords+1]))
            
            
            # lq = min(len(s1),len(s2)) + 10
            # ls = max(len(s1),len(s2))
            
            lscore = 1/len(s2) # This ensures that we always choose the shorter suggestion over the longer one if both have the same score otherwise
            
            # if the two strings are close enough already no need to perform further calculations
            if lvs.distance(s1[:lq],s2[:lq]) < 3 or osa.distance(s1[:lq],s2[:lq]) < 2:
                return 50
            
            score = 0.5*lvs.similarity(s1[:lq],s2[:lq],score_cutoff=score_cutoff) +\
                    0.35*osa.similarity(s1[:lq],s2[:lq],score_cutoff=score_cutoff) +\
                    0.15*( ( fuzz.token_set_ratio(s1[:lq],s2[:lq],score_cutoff=((score_cutoff/lq)*100) ) / 100 ) * lq)
                    # 0.1*rep1.similarity(rep2)*10
            
            if score < score_cutoff:
                return 0
            # if s1[:lq] == s2[:lq]:
            #     print(s1[:lq],s2[:lq],score)
            score += lscore
            return score
        
        # processor=lambda a: a[:len(q)+5],
        best_suggestions = process.extract(q,list(self.sents_data),scorer=scorer_fn,limit=top_k,score_cutoff=score_cutoff)
        for i in range(len(best_suggestions)):
            # print(best_suggestions[i])
            if any([best_suggestions[i][0].endswith(x) for x in ['.','?']]):
                best_suggestions[i] = best_suggestions[i][0]
            elif any([best_suggestions[i][0].lower().startswith(x) for x in ['what','which','how','who']]):
                best_suggestions[i] = best_suggestions[i][0] + '?'
            elif best_suggestions[i][0].lower().startswith('list'):
                best_suggestions[i] = best_suggestions[i][0] + '.'
            else:
                best_suggestions[i] = best_suggestions[i][0] + '.?'
            best_suggestions[i] = best_suggestions[i][0].upper() + best_suggestions[i][1:]
        return {'suggestions':best_suggestions,'heading':"",'rows':[]}
    
    def add_data(self, q: str, write_to_file: bool = False):
        self.sents_data.add(q)
        if len(self.sents_data_path) > 0 and write_to_file:
            if self.sents_data_path.endswith('.pkl'):
                with open(self.sents_data_path, 'wb') as f:
                    pickle.dump(list(self.sents_data), f)
            elif self.sents_data_path.endswith('.txt'):
                with open(self.sents_data_path, 'w') as f:
                    f.write('\n'.join(self.sents_data))

