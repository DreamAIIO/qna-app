import csv
import json
import re
from functools import partial
from typing import (Any, AsyncIterator, Callable, Dict, Hashable, Iterator,
                    List, Optional, Tuple, Union)

import numpy as np
import pandas as pd
import rapidfuzz as rf
from langchain.chains.base import Chain
from langchain.chains.graph_qa.cypher import construct_schema, extract_cypher
from langchain.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema
from langchain.chains.graph_qa.prompts import CYPHER_QA_PROMPT
from langchain.chains.llm import LLMChain
from langchain.prompts import (BasePromptTemplate, ChatPromptTemplate,
                               HumanMessagePromptTemplate, MessagesPlaceholder,
                               PromptTemplate, SystemMessagePromptTemplate)
# from langchain_community.chat_models import ChatOpenAI
# from langchain.graphs.neo4j_graph import Neo4jGraph
from langchain_community.graphs.neo4j_graph import Neo4jGraph
# from langchain.callbacks.manager import (AsyncCallbackManager,
#                                          AsyncCallbackManagerForChainRun,
#                                          CallbackManager,
#                                          CallbackManagerForChainRun)
from langchain_core.callbacks.manager import (AsyncCallbackManager,
                                              AsyncCallbackManagerForChainRun,
                                              CallbackManager,
                                              CallbackManagerForChainRun)
# from langchain.base_language import BaseLanguageModel
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai import ChatOpenAI
from pydantic import Field

CYPHER_GENERATION_SYSTEM = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Also use Neo4j's Fulltext Search to search and return the 'number' of any extracted entity using its name.
Use the given schema and examples as a guide. Ignore any relationship types and properties not in the schema
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

Examples:
# What is the average price of items from the bakery category
```
MATCH (i:Item)
WHERE i.category =~ "(?i).*bakery.*"
RETURN AVG(i.price) AS average_item_price;
```
# What is the item info of perdue chicken breast tenders.
```
CALL {{
    CALL db.index.fulltext.queryNodes('namesAndDescriptions',"perdue chicken breast tenders OR perdue~ chicken~ breast~ tenders~") YIELD node, score
    WHERE node:Item AND score > 2.0 AND apoc.text.distance(node.brand_name,"perdue") <= 0.5*SIZE(node.brand_name)
    RETURN collect(node)[0..10] AS items
}}
UNWIND items AS i
RETURN i.number AS item_number, i.brand_name AS item_brand_name, i.description AS item_description, i.ven_item_number AS vendor_item_number, i.upc AS item_upc, i.price AS item_price, i.po_cost AS item_po_cost, i.po_cost_unit AS item_po_cost_unit, i.freight AS item_freight, i.po_cost_freight AS item_po_cost_freight, i.store_cost AS item_store_cost, i.everyday_margin AS item_everyday_margin, i.category AS item_category, i.itype AS item_type, i.item_pack AS item_pack LIMIT 500;
```
# List all customers that bought turkey breast oven in May 2021.
```
CALL {{
    CALL db.index.fulltext.queryNodes('namesAndDescriptions',"turkey breast oven OR turkey~ breast~ oven~") YIELD node, score
    WHERE node:Item AND score > 2.0
    RETURN collect(node)[0..10] AS items
}}
UNWIND items AS i
MATCH (i:Item)-[:RECORDED_ITEM]-(r:Record)
MATCH (c:Customer)-[:RECORDED_CUSTOMER]-(r:Record)
MATCH (c:Customer)-[:BOUGHT_ITEM]-(i:Item)
WHERE r.record_date.month == 5 AND r.record_data.year == 2021
RETURN c.number AS customer_number, c.name AS customer_name, v.number AS vendor_number, v.name AS vendor_name LIMIT 400;
```
# List items sold by perdue farms(froz) and bought by customer 95452006?
```
CALL {{
    CALL db.index.fulltext.queryNodes('namesAndDescriptions',"perdue farms froz OR perdue~ farms~ froz~") YIELD node, score
    WHERE node:Vendor AND score > 2.0
    RETURN collect(node)[0..10] AS vendors
}}
UNWIND vendors AS v
MATCH (v:Vendor)-[:SOLD_ITEM]-(i:Item)
MATCH (c:Customer {{number: '95452006'}})-[:BOUGHT_ITEM]-(i:Item)
RETURN DISTINCT i.number AS item_number, i.name AS item_name, v.number AS vendor_number, v.name AS vendor_name LIMIT 400;
```
# List the top 10 vendors that family foods bought from.
```
CALL {{
    CALL db.index.fulltext.queryNodes('namesAndDescriptions',"family foods OR family~ foods~") YIELD node, score
    WHERE node:Customer AND score > 2.0
    RETURN collect(node)[0..10] AS customers
}}
UNWIND customers AS c
MATCH (v:Vendor)-[:RECORDED_VENDOR]-(r:Record)
MATCH (c:Customer)-[:RECORDED_CUSTOMER]-(r:Record)
RETURN v.number AS vendor_number, v.name AS vendor_name, c.number AS customer_number, c.name AS customer_name, SUM(r.record_cases) AS sum_record_cases
ORDER BY SUM(r.record_cases) LIMIT 10;
```
# How many cases of aunt butchies cheesecake cone were bought by harvest market in 2022?
```
CALL {{
    CALL db.index.fulltext.queryNodes('namesAndDescriptions', "aunt butchies cheesecake cone OR aunt~ butchies~ cheesecake~ cone~") YIELD node, score
    WHERE node:Item AND score > 2.0 AND apoc.text.distance(node.brand_name,"aunt butchies") <= 0.5*SIZE(node.brand_name)
    RETURN collect(node)[0..10] AS items
}}
CALL {{
    CALL db.index.fulltext.queryNodes('namesAndDescriptions',"harvest market OR harvest~ market~") YIELD node, score
    WHERE node:Customer AND score > 2.0
    RETURN collect(node)[0..10] AS customers
}}
UNWIND items AS i
UNWIND customers AS c
MATCH (r:Record)-[:RECORDED_ITEM]-(i:Item)
MATCH (r:Record)-[:RECORDED_CUSTOMER]-(c:Customer)
WHERE r.record_data.year = 2022
RETURN SUM(r.record_cases) AS sum_record_cases, i.number AS item_number, i.name AS item_name, c.number AS customer_number, c.name AS customer_name LIMIT 400;
```
# What is the store cost and margin of the top 30 items that customer winey cow purchased from vendor 5357.
```
CALL {{
    CALL db.index.fulltext.queryNodes('namesAndDescriptions',"winey cow OR winey~ cow~") YIELD node, score
    WHERE node:Customer AND score > 2.0
    RETURN collect(node)[0..10] AS customers
}}
UNWIND customers AS c
MATCH (r:Record)-[:RECORDED_CUSTOMER]-(c:Customer)
MATCH (r:Record)-[:RECORDED_VENDOR]-(v:Vendor {{number: 5357}})
MATCH (r:Record)-[:RECORDED_ITEM]-(i:Item)
RETURN i.store_cost_cs AS item_store_cost, i.zone8_margin AS item_zone8_margin, i.number AS item_number, i.name AS item_name, v.number AS vendor_number, v.name AS vendor_name, c.number AS customer_number, c.name AS customer_name, SUM(r.record_cases) AS sum_record_cases
ORDER BY sum_record_cases DESC LIMIT 30;
```
"""
# vendor 5357: hoff's bakery

CYPHER_GENERATION_SYSTEM2 = """You are a data science engineer able to generate cypher statement to query a graph database using the given instructions, graph schema and some examples.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
A list of numbers for the entities will be provided as parameters. These numbers always correspond to the `number` property. When provided, always use the `number` property to search for entities. 
Do NOT use any of the following properties: `number`, `name`, `description` inside the WHERE clause.
Use case-insensitive fuzzy string matching for filtering string properties that aren't `name` or `description`.
Always use `SUM(r.record_cases)` as the default metric for sorting - that includes `(r:Record)`.
When `top` is mentioned in the query, always use r:Record inside the Cypher statement.
When asked to provide `info`, return all the properties of the entity.
Always return DISTINCT results and use COUNT(DISTINCT()) for counting.

Schema:
{schema}
The property Customer.state contains two letter state codes for a state e.g. "OH" for "Ohio"

Examples:
Question: Which vendor sells cheesecake cone almond?
Parameters: cheesecake cone almond: inos
Cypher: ```
UNWIND $inos AS ino
MATCH (v:Vendor)-[:SOLD_ITEM]-(i:Item {{number: ino}})
RETURN DISTINCT v.number AS vendor_number, v.name AS vendor_name, i.number AS item_number, i.name AS item_number;
```
Question: What is the price and category of items sold by perdue farms and bought by family foods.
Parameters: perdue farms: vnos, family foods: cnos
Cypher: ```
UNWIND $vnos AS vno
UNWIND $cnos AS cno
MATCH (v:Vendor {{number: vno}})-[:SOLD_ITEM]-(i:Item)
MATCH (c:Customer {{number: cno}})-[:BOUGHT_ITEM]-(i:Item)
RETURN DISTINCT i.price AS item_price, i.category AS item_category, i.number AS item_number, i.name AS item_name, v.number AS vendor_number, v.name AS vendor_name, c.number AS customer_number, c.name AS customer_name;
```
Question: How many cases of aunt butchies cheesecake cone were bought by harvest market in 2022?
Parameters: aunt butchies cheesecake cone: inos, harvest market: cnos
Cypher: ```
UNWIND $inos AS ino
UNWIND $cnos AS cno
MATCH (r:Record)-[:RECORDED_ITEM]-(i:Item {{number: ino}})
MATCH (r:Record)-[:RECORDED_CUSTOMER]-(c:Customer {{number: cno}})
WHERE r.record_data.year = 2022
RETURN DISTINCT SUM(r.record_cases) AS sum_record_cases, i.number AS item_number, i.name AS item_name, c.number AS customer_number, c.name AS customer_name;
```
Question: List the top 30 items that customer winey cow purchased from vendor 5357.
Parameters: vendor 5357: vnos, customer winey cow: cnos
Cypher: ```
UNWIND $vnos AS vno
UNWIND $cnos AS cno
MATCH (r:Record)-[:RECORDED_CUSTOMER]-(c:Customer {{number: cno}})
MATCH (r:Record)-[:RECORDED_VENDOR]-(v:Vendor {{number: vno}})
MATCH (r:Record)-[:RECORDED_ITEM]-(i:Item)
RETURN DISTINCT i.number AS item_number, i.name AS item_name, v.number AS vendor_number, v.name AS vendor_name, c.number AS customer_number, c.name AS customer_name, SUM(r.record_cases) AS sum_record_cases
ORDER BY sum_record_cases DESC LIMIT 30;
```
Question: List all records with customer winey cow and vendor hoff's bakery.
Parameters: customer winey cow: cnos, vendor hoff's bakery: vnos
Cypher: ```
UNWIND $cnos AS cno
UNWIND $vnos AS vno
MATCH (r:Record)-[:RECORDED_CUSTOMER]-(c:Customer {{number: cno}})
MATCH (r:Record)-[:RECORDED_VENDOR]-(v:Vendor {{number: vno}})
RETURN DISTINCT r.record_date AS record_date, r.record_cases AS record_cases, r.record_weight AS record_weight, c.number AS customer_number, c.name AS customer_name, v.number AS vendor_number, v.name AS vendor_name LIMIT 400;
```
Question: List all vendors that had sales handled by sales rep dean bolt in the sales region mi-west.
Parameters: sales rep dean bolt: srepnos, sales region mi-west: sregnos
Cypher: ```
UNWIND $srepnos AS srepno
UNWIND $sregnos AS sregno
MATCH (r:Record)-[:RECORDED_SALES_REP]-(srep: SalesRep {{number: srepno}})
MATCH (r:Record)-[:RECORDED_SALES_REGION]-(sreg: SalesRegion {{sregno}})
MATCH (r)-[:RECORDED_VENDOR]-(v:Vendor)
RETURN DISTINCT v.number AS vendor_number, v.name AS vendor_name, srep.number AS sales_rep_number, srep.name AS sales_rep_name, sreg.number AS sales_region_number, sreg.description AS sales_region_name;
```
Question: List the top 50 customers that bought items from the deli category.
Parameters: 
Cypher: ```
MATCH (r:Record)-[:RECORDED_CUSTOMER]-(c:Customer)
MATCH (r:Record)-[:RECORDED_ITEM]-(i:Item)
MATCH (c)-[:BOUGHT_ITEM]-(i)
WHERE i.category =~ "(?i).*deli.*"
RETURN DISTINCT c.number AS customer_number, c.name AS customer_name, i.number AS item_number, i.name AS item_name, i.category AS item_category, SUM(r.record_cases) AS sum_record_cases;
ORDER BY sum_record_cases LIMIT 50;
```
Question: How many items from the bakery category does aunt butchies sell?
Parameters: aunt butchies: vnos
Cypher: ```
UNWIND $vnos AS vno
MATCH (v:Vendor {{number: vno}})-[:BOUGHT_ITEM]-(i:Item)
WHERE i.category =~ "(?i).*bakery.*"
RETURN COUNT(DISTINCT(i.number)) AS item_count, i.category, v.number AS vendor_number, v.name AS vendor_name;
```
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Make sure to double-check that the query conforms to the given schema."""

# QUERY_INPUT_HUMAN = """Question: {question}
# Cypher:"""

QUERY_INPUT_HUMAN = """Question: {question}
Parameters: {params}
Cypher:"""

ENTITY_EXTRACTION_TEMPLATE = """Given the input text, extract a comma-separated list of entities with their types: <entity_type> <entity>.
An entity is the name/number of vendor, customer, item, sales rep or sales region. Ignore any other entities.
Allowed Types: `vendor, item, customer, sales rep, sales region`.

Examples:
```
Input: how should I sell cheesecake cone to martins
Entities: [item cheesecake cone, customer martins]
Input: which item is more expensive: perdue chicken breast or troyer turkey breast
Entities: [item perdue chicken breast, item troyer turkey breast]
Input: list the items that had sales handled by dean bolt in the sales region mi-west
Entities: [sales rep dean bolt, sales region mi-west]
Input: what is the info of items bought by customer 81423002 and sold by vendor 4993
Entities: [customer 81423002, vendor 4993]
Input: list items from the bakery category that have po cost greater than $10.
Entities: []
Input: list the 10 most expensive bakery items based on po cost with price greater than $10
Entities: []
Input: what is the info of the bakery items sold by aunt butchies that have po cost greater than 20
Entities: [vendor aunt butchies]
Input: list items sold by smithfield and bought by customers that are based in the state oh
Entities: [vendor smithfield]
```

Input: {input}
Entities:
"""


def to_titlecase(text: str, all_upper: List[str] = ["srp", "upc"]) -> str:
    if text in all_upper:
        return text.upper()
    return text[0].upper() + text[1:].lower()


def is_number(text: str) -> bool:
    return all([text[i] in "0123456789." for i in range(len(text))])


def is_snumber(text: str) -> bool:
    dlen = 0
    for t in text:
        if t in "0123456789":
            dlen += 1
    return len(text) - dlen <= 2  # if there are no more than 2 non-digit characters


def extract_number(text: str) -> str:
    return re.findall(r"\b\d+\b", text)


def is_punctuation(t: str) -> bool:
    return t in [
        "-",
        ".",
        ",",
        "\\",
        "?",
        "/",
        "'",
        '"',
        ":",
        ";",
        "(",
        ")",
        "!",
        "`",
        "",
    ]


def format_float(f: Union[int, float], form: str) -> str:
    try:
        f = float(f)
        if f >= 0:
            return form.format(f)
        return "Not Available"
    except:
        return f


def convertToTitleCase(colname: str, all_upper: List[str] = ["srp", "upc"]) -> str:
    alias = ""
    for w in re.split("_| |-|\.", colname):
        if w == 'i':
            w = 'Item'
        elif w == 'v':
            w = 'Vendor'
        elif w == 'c':
            w = 'Customer'
        elif w == 'srep':
            w = 'Sales Rep'
        elif w == 'sreg':
            w = 'Sales Region'
        elif w == 'r':
            w = 'Record'
        else:
            w = to_titlecase(w, all_upper)
        if not w in alias:
            alias += w + " "
    return alias[:-1]


class EntityExtractor:

    

    def __init__(
        self, ner_modelpath="Bert_NER", tokenizer_path="bert-base-uncased", **kwargs
    ):
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        from nltk.corpus import stopwords
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, is_fast=True)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(
            ner_modelpath
        ).to(device)

        self.stopwords = set(stopwords.words("english"))
        self.stopwords.discard("or")
        self.stopwords.discard("and")
        self.stopwords.discard(",")
        self.slots_labels = [
            "O",
            "[PAD]",
            "[UNK]",
            "B-customer",
            "I-customer",
            "B-customer_prop",
            "I-customer_prop",
            "B-delivery_date",
            "I-delivery_date",
            "B-item",
            "I-item",
            "B-item_prop",
            "I-item_prop",
            "B-order_date",
            "I-order_date",
            "B-po_number",
            "I-po_number",
            "B-po_prop",
            "I-po_prop",
            "B-prop_of_customer",
            "I-prop_of_customer",
            "B-prop_of_item",
            "I-prop_of_item",
            "B-prop_of_po",
            "I-prop_of_po",
            "B-prop_of_record",
            "I-prop_of_record",
            "B-prop_of_salesregion",
            "I-prop_of_salesregion",
            "B-prop_of_salesrep",
            "I-prop_of_salesrep",
            "B-prop_of_vendor",
            "I-prop_of_vendor",
            "B-record_date",
            "I-record_date",
            "B-record_prop",
            "I-record_prop",
            "B-sales_region",
            "I-sales_region",
            "B-sales_rep",
            "I-sales_rep",
            "B-top_bottom",
            "I-top_bottom",
            "B-vendor",
            "I-vendor",
            "B-vendor_prop",
            "I-vendor_prop",
        ]

    def preprocess_pred(self, sent: str, device, max_length: int = 45):
        tokenized_input = self.tokenizer(
            sent,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        ).to(device)
        return (
            tokenized_input["input_ids"],
            tokenized_input["attention_mask"],
            tokenized_input["token_type_ids"],
        )

    def _extract_all_entities(self, input_text: str):
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.ner_model.eval()
        input_ids, input_mask, token_type_ids = self.preprocess_pred(input_text, device=device)
        with torch.no_grad():
            outputs = self.ner_model(input_ids, input_mask, token_type_ids)
        slots_pred = np.argmax(outputs.logits.detach().cpu().numpy(), axis=2)[0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().cpu())
        word_tokens, lab_tokens = [], []
        for token, pred_idx in zip(tokens, slots_pred):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:  # ignore these tokens
                continue
            if token.startswith("##"):
                word_tokens[-1] = word_tokens[-1] + token[2:]
            else:
                lab_tokens.append(self.slots_labels[pred_idx])
                word_tokens.append(token)

        assert len(word_tokens) == len(
            lab_tokens
        ), "Entities could not be extracted properly!"

        entities = []
        i = 0
        while i < len(word_tokens):
            if lab_tokens[i][0] == "B":
                entity = word_tokens[i]
                ent_type = lab_tokens[i][2:]
                j = i + 1
                while j < len(word_tokens) and (
                    lab_tokens[j][0] == "I"
                    or (lab_tokens[j][0] == "B" and lab_tokens[j][2:] == ent_type)
                ):
                    if not is_punctuation(word_tokens[j]) and (
                        j != 0 and not (is_punctuation(word_tokens[j - 1]))
                    ):
                        entity += " "
                    entity += word_tokens[j]
                    if ent_type != lab_tokens[j][2:]:
                        print(
                            f"Unexpected token found: {ent_type} != {lab_tokens[j][2:]}"
                        )
                        # ent_type = lab_tokens[j][2:]
                        # entity = ' '.join(entity.split(' ')[1:]) # Drop the first word
                    j += 1
                entity = " ".join(
                    [w for w in entity.split() if w not in self.stopwords]
                )
                ents = re.split(r" [or|and|,] ", entity, flags=re.I)
                for ent in ents:
                    entities.append({"text": ent, "tag": ent_type})
                # entities.append(
                #     {"text": entity, "tag": ent_type, "start": i, "end": j - 1}
                # )
                i = j
            else:
                i += 1
        return entities

    async def _aextract_all_entities(self, input_text: str):
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ner_model.eval()
        input_ids, input_mask, token_type_ids = self.preprocess_pred(input_text, device=device)
        with torch.no_grad():
            outputs: torch.Tensor = self.ner_model(
                input_ids, input_mask, token_type_ids
            )
        slots_pred = np.argmax(outputs.logits.detach().cpu().numpy(), axis=2)[0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().cpu())
        word_tokens, lab_tokens = [], []
        for token, pred_idx in zip(tokens, slots_pred):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:  # ignore these tokens
                continue
            if token.startswith("##"):
                word_tokens[-1] = word_tokens[-1] + token[2:]
            else:
                lab_tokens.append(self.slots_labels[pred_idx])
                word_tokens.append(token)

        assert len(word_tokens) == len(
            lab_tokens
        ), "Entities could not be extracted properly!"

        entities = []
        i = 0
        while i < len(word_tokens):
            if lab_tokens[i][0] == "B":
                entity = word_tokens[i]
                ent_type = lab_tokens[i][2:]
                j = i + 1
                while j < len(word_tokens) and (
                    lab_tokens[j][0] == "I"
                    or (lab_tokens[j][0] == "B" and lab_tokens[j][2:] == ent_type)
                ):
                    if not is_punctuation(word_tokens[j]) and (
                        j != 0 and not (is_punctuation(word_tokens[j - 1]))
                    ):
                        entity += " "
                    entity += word_tokens[j]
                    if ent_type != lab_tokens[j][2:]:
                        print(
                            f"Unexpected token found: {ent_type} != {lab_tokens[j][2:]}"
                        )
                        # ent_type = lab_tokens[j][2:]
                        # entity = ' '.join(entity.split(' ')[1:]) # Drop the first word
                    j += 1
                entity = " ".join(
                    [w for w in entity.split() if w not in self.stopwords]
                )
                ents = re.split(r" [or|and|,] ", entity, flags=re.I)
                for ent in ents:
                    entities.append({"text": ent, "tag": ent_type})
                # entities.append(
                #     {"text": entity, "tag": ent_type, "start": i, "end": j - 1}
                # )
                i = j
            else:
                i += 1
        return entities

    def extract_entities_lazy(self, input_text: str) -> Iterator[Dict[str, str]]:
        entities = self._extract_all_entities(input_text)
        for ent in entities:
            if ent["tag"].lower() in [
                "item",
                "customer",
                "vendor",
                "sales_rep",
                "sales_region",
            ]:
                yield ent

    def extract_entities(self, input_text: str) -> List[Dict[str, str]]:
        entities = self._extract_all_entities(input_text)
        return [
            ent
            for ent in entities
            if ent["tag"].lower()
            in ["item", "customer", "vendor", "sales_rep", "sales_region"]
        ]

    async def async_extract_entities_lazy(
        self, input_text: str
    ) -> AsyncIterator[Dict[str, str]]:
        entities = await self._aextract_all_entities(input_text)
        for ent in entities:
            if ent["tag"].lower() in [
                "item",
                "customer",
                "vendor",
                "sales_rep",
                "sales_region",
            ]:
                yield ent

    async def async_extract_entities(self, input_text: str) -> List[Dict[str, str]]:
        entities = self._aextract_all_entities(input_text)
        return [
            ent
            for ent in entities
            if ent["tag"].lower()
            in ["item", "customer", "vendor", "sales_rep", "sales_region"]
        ]


# class QueryClassifier:
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EntityExtractor2:
    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        prompt: str = ENTITY_EXTRACTION_TEMPLATE,
        ner_modelpath="Bert_NER",
        tokenizer_path="bert-base-uncased",
    ):
        if llm is None:
            llm = ChatOpenAI(temperature=0)
        prompt = PromptTemplate.from_template(prompt)
        # ent_extractor = EntityExtractor(ner_modelpath=ner_modelpath, tokenizer_path=tokenizer_path)
        self.entity_extraction_chain = LLMChain(llm=llm, prompt=prompt)

    @staticmethod
    def extract_output(text: str) -> str:
        return (
            extract_cypher(re.split(r"Entities: ", text, flags=re.I)[-1])
            .strip("[]")
            .strip()
        )
        """
        match = re.findall(r"Entities: (.+)", text, flags=re.I)
        if len(match) > 0:
            match = match[-1].strip(': ')
            if len(match) == 0:
                return text
        return text
        """

    def extract_entities_lazy(self, text: str) -> Iterator[Dict[str, str]]:
        text = text.lower().strip(".?")
        entities = self.entity_extraction_chain.invoke(input=text, return_only_outputs=True)[self.entity_extraction_chain.output_keys[0]]
        # print(entities)
        entities = self.extract_output(entities)
        # print(entities)
        entities = [ent.strip() for ent in entities.split(",")]
        # for ent in self.ent_extractor.extract_entities_lazy(entities):
        for ent in entities:
            if len(ent) == 0:
                continue
            ent_type = re.findall(
                r"(sales rep|sales region|item|customer|vendor) (.*)", ent, flags=re.I
            )
            # print(ent, ent_type)
            if len(ent_type) > 0:
                ent_text = ent_type[0][1].strip()
                ent_type = ent_type[0][0].strip()
                # print(ent_text, ent_type)
                che = re.search(
                    r"\b(category|item|brand|price|po|cost|unit|srp|store|address|city|state|vendor|customer|record|sale?|rep|region|bakery|deli|meat|seafood|dairy)s?\b",
                    ent_text,
                    re.I,
                )

                # print(che, che is None)
                if ent != ent_type and che is None:
                    ent_type = ent_type.lower().replace(" ", "_")
                    yield {"tag": ent_type, "text": ent_type + " " + ent_text}

    async def async_extract_entities_lazy(self, text: str) -> AsyncIterator[Dict[str, str]]:
        text = text.lower().strip(".?")
        entities = await self.entity_extraction_chain.arun(input=text)
        # print(entities)
        entities = self.extract_output(entities)
        # print(entities)
        entities = [ent.strip() for ent in entities.split(",")]
        # for ent in self.ent_extractor.extract_entities_lazy(entities):
        for ent in entities:
            if len(ent) == 0:
                continue
            ent_type = re.findall(
                r"(sales rep|sales region|item|customer|vendor) (.*)", ent, flags=re.I
            )
            # print(ent, ent_type)
            if len(ent_type) > 0:
                ent_text = ent_type[0][1].strip()
                ent_type = ent_type[0][0].strip()
                # print(ent_text, ent_type)
                che = re.search(
                    r"\b(category|item|brand|price|po|cost|unit|srp|store|address|city|state|vendor|customer|record|sale?|rep|region|bakery|deli|meat|seafood|dairy)s?\b",
                    ent_text,
                    re.I,
                )

                # print(che, che is None)
                if ent != ent_type and che is None:
                    ent_type = ent_type.lower().replace(" ", "_")
                    yield {"tag": ent_type, "text": ent_type + " " + ent_text}

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        return list(self.extract_entities_lazy(text))
        # entities = self.entity_extraction_chain(text)
        # entities = [ent.strip() for ent in entities.split(',')]
        # return self.ent_extractor.extract_entities(entities)

    async def async_extract_entities(self, text: str) -> List[Dict[str, str]]:
        return list(await self.async_extract_entities_lazy(text))


class DaiNeo4jGraph(Neo4jGraph):
    """
    Inherit and overload Neo4jGraph's query method with custom query method
    that allows returning the result as a pandas dataframe
    """

    import neo4j

    def __init__(
        self,
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "neo4j",
    ) -> None:
        import os

        """Create a new Neo4j graph wrapper instance."""
        
        url = os.environ.get("NEO4J_URI", url) #get_from_env("url", "NEO4J_URI", url)
        username = os.environ.get("NEO4J_USERNAME", username) #get_from_env("username", "NEO4J_USERNAME", username)
        password = os.environ.get("NEO4J_PASSWORD", password) #get_from_env("password", "NEO4J_PASSWORD", password)
        database = os.environ.get("NEO4J_DATABASE", database) #get_from_env("database", "NEO4J_DATABASE", database)
        
        if password is None or username is None:
            auth = None
        elif username is not None and password is not None:
            auth = (username, password)
        else:
            print(username, password)
            auth = None
        self._driver = self.neo4j.GraphDatabase.driver(url, auth=auth)
        self._database = database
        self.schema: str = ""
        self.structured_schema: Dict[str, Any] = {}
        # Verify connection
        try:
            self._driver.verify_connectivity()
        except self.neo4j.exceptions.ServiceUnavailable:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the url is correct"
            )
        except self.neo4j.exceptions.AuthError:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the username and password are correct"
            )
        # Set schema
        try:
            self.refresh_schema()
        except self.neo4j.exceptions.ClientError:
            raise ValueError(
                "Could not use APOC procedures. "
                "Please ensure the APOC plugin is installed in Neo4j and that "
                "'apoc.meta.data()' is allowed in Neo4j configuration "
            )

    @staticmethod
    def node_to_dict(node: neo4j.graph.Node, ignore_keys: List = []):
        label = list(node.labels)[0].lower()
        d = {}
        for k, v in node.items():
            nlabel = f"{label}_{k}"
            if not nlabel in ignore_keys:
                d[nlabel] = v
        return d

    def query(
        self,
        query: str,
        params: dict = {},
        return_dframe=False,
        return_dict=False,
        remove_duplicates=True,
    ) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """Query Neo4j database."""
        # from neo4j.exceptions import CypherSyntaxError

        with self._driver.session(database=self._database) as session:
            try:
                result = session.run(query, params)
                if return_dict or return_dframe:
                    result = pd.DataFrame(
                        [r.values() for r in result], columns=result.keys()
                    )
                    if len(result) == 0:
                        return result
                    for c in result.columns:
                        if isinstance(result.loc[0, c], self.neo4j.graph.Node):
                            cdict = result[c].apply(
                                partial(self.node_to_dict, ignore_keys=result.columns)
                            )
                            del result[c]
                            if len(cdict) == 0 or len(cdict[0]) == 0:
                                continue

                            new_columns_df = pd.DataFrame.from_records(cdict)
                            # result = result.join(new_columns_df, how="outer")
                            result = pd.concat([result, new_columns_df], axis=1)
                        elif isinstance(result.loc[0, c], list):
                            result[c] = result[c].apply(lambda a: ',, '.join(a))
                        elif isinstance(result.loc[0, c], str) and '[' in result.loc[0,c]:
                            result[c] = result[c].apply(lambda a: a.strip(' []').strip())
                    # result[result.select_dtypes("object").columns] = result.select_dtypes("object").fillna("NA")
                    # result[result.select_dtypes("float").columns] = result.select_dtypes("float").fillna(-1)
                    # result[result.select_dtypes("int").columns] = result.select_dtypes("int").fillna(-1)
                    result = result.fillna("Not Available")
                    # print(result.columns, len(result))
                    if remove_duplicates:
                        cols = [
                            c
                            for c in result.columns
                            if isinstance(result.loc[0, c], Hashable)
                        ]
                        if len(cols) > 0:
                            result.drop_duplicates(subset=cols, inplace=True)
                    if return_dframe:
                        return result.iloc[:500]
                    else:
                        return result.to_dict(orient="records")
                else:
                    return [r.data() for r in result]
            except self.neo4j.exceptions.ClientError as ce:
                raise ValueError(f"Generated Cypher Statement is not valid\n{ce}")

    def search_index_lazy(
        self,
        entities: List[Dict[str, str]],
        top_k: int = 10,
        max_distance: float = 2.0,
        max_difference: float = 5.0,
    ) -> Iterator[Tuple[str, Union[str, int]]]:
        for ent in entities:
            if isinstance(ent, dict):
                try:
                    query, prop = (
                        ent.get("text", ""),
                        ent.get("tag", ""),
                    )
                except Exception as e:
                    print(ent, e)
                    raise Exception(
                        "`entities` must be a list of dictionaries with keys: ['text','tag']"
                    )
            elif isinstance(ent, tuple) or isinstance(ent, list):
                query, prop = ent[0], ent[1]
            else:
                raise Exception(
                    f"A single entity must be a list of dictionaries, tuples or lists. Not {type(ent)}"
                )
            if "vendor" in prop.lower():
                prop = "Vendor"
            elif "customer" in prop.lower():
                prop = "Customer"
            elif "item" in prop.lower():
                prop = "Item"
            elif "sales" in prop.lower() and "rep" in prop.lower():
                prop = "SalesRep"
            elif "sales" in prop.lower() and "region" in prop.lower():
                prop = "SalesRegion"
            elif prop == "":
                continue
            else:
                raise Exception(f"Unknown class: {prop}")

            query = re.sub(r"\.|\?|!", "", query).lower()
            query = re.sub(r"\)|\(|-|\\|/", " ", query)
            query = re.sub(
                r"\b(item|product|vendor|seller|customer|buyer|purchase|order|record|sale|rep|no|number|representative|region|number|reg|no)s?\b",
                "",
                query,
                flags=re.I,
            ).strip()
            if query == "":
                # yield (prop, [])
                continue  # ignore if query becomes empty after removal of keywords
            if is_number(query):
                if prop != "Customer":
                    query = int(query)
                # else:
                #     query = f"'{query}'"
                yield (prop, [query])
                continue
            words = [x for x in query.split(" ") if x != ""]
            fquery = "~ ".join(words)
            squery = f"{fquery}~ OR {query}"
            cquery = f"""CALL db.index.fulltext.queryNodes('namesAndDescriptions',$sterm) YIELD node,score
WHERE node:{prop} AND score > {max_distance}
RETURN node.number AS number, node.name AS name, node.brand_name AS brand, node.description AS description,score LIMIT {top_k};"""

            ids_dists = self.query(
                cquery,
                params={"sterm": squery},
                return_dframe=True,
                remove_duplicates=False,
            )
            if len(ids_dists) == 0:
                print(
                    f"'{query}' entity of type '{prop}' was not found in KG! Please check for spelling errors or try to add more description/detail!"
                )
                yield (prop, [])
                continue
                # return f"'{query}' entity of type '{prop}'  was not found in KG! Please check for spelling errors or try to add more description/detail!"

            if prop == "Item" and (
                query.lower().startswith(ids_dists.loc[0, "brand"].lower())
                or rf.fuzz.partial_token_ratio(query.lower(), ids_dists.loc[0, "brand"])
                > 90
            ):
                brand = ids_dists.loc[0, "brand"]
            else:
                brand = ""

            ids = [ids_dists.loc[0, "number"]]
            i = 1
            while (
                i < len(ids_dists)
                and (ids_dists.loc[i - 1, "score"] - ids_dists.loc[i, "score"])
                <= max_difference
            ):
                if brand == "" or ids_dists.loc[i, "brand"] == brand:
                    ids.append(ids_dists.loc[i, "number"])
                i += 1
            yield (prop, ids)

    def search_index(
        self,
        entities: List[Dict[str, str]],
        top_k: int = 10,
        max_distance: float = 2.0,
        max_difference: float = 5.0,
    ) -> List[Tuple[str, Union[str, int]]]:
        return list(
            self.search_index_lazy(entities, top_k, max_distance, max_difference)
        )

    def vector_search(
        self,
        query: Optional[str] = None,
        query_emb: Optional[Any] = None,
        embed_fn: Optional[Callable] = None,
    ) -> str:
        if query is not None and embed_fn is not None:
            query_emb = embed_fn(query)
        if query_emb is not None:
            vector_search_query = """CALL db.index.vector.queryNodes($emb)"""
            # TODO: Complete vector index search using Neo4j's newest feature
            raise NotImplementedError()
        else:
            raise ValueError(
                "Either provide both `query` and `embed_fn` or just `query_emb`"
            )


class DaiGraphCypherQAChain(Chain):
    """
    Chain for question-answering against a graph by generating Cypher statements.

    It either returns a table of answers or a well-reasoned free text answer.
    """

    graph: DaiNeo4jGraph = Field(exclude=True)
    cypher_generation_chain: LLMChain
    qa_chain: LLMChain
    entity_extractor: Optional[EntityExtractor2] = None
    # query_classifier: Any  # A few-shot classifier to distinguish between input queries
    # query_classes: List[str] = [
    #     "table",
    #     "text",
    # ]  # whether to return a table or a free text response
    graph_schema: str
    input_key: str = "query"  #: :meta private:
    filters_key: str = "filters" #: :meta private:
    output_heading_key: str = "heading" #: :meta private:
    output_suggs_key: str = "suggestions" #: :meta private:
    output_rows_key: str = "rows" #: :meta private:
    output_text_result_key: str = "llm_response" #: :meta private:
    top_k: int = 100
    """Number of results to return from the query"""
    cypher_query_corrector: Optional[CypherQueryCorrector] = None
    """Optional cypher validation tool"""

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.

        :meta private:
        """
        _output_keys = [
            self.output_heading_key,
            self.output_rows_key,
            self.output_suggs_key,
            self.output_text_result_key,
        ]
        return _output_keys

    @classmethod
    def create_cypher_prompt(
        cls,
        system_message: str = CYPHER_GENERATION_SYSTEM,
        human_message: str = QUERY_INPUT_HUMAN,
        input_variables: Optional[List[str]] = None,
    ) -> BasePromptTemplate:
        if input_variables is None:
            input_variables = ["schema", "question", "chat_history"]
        messages = [
            SystemMessagePromptTemplate(prompt=system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate(human_message),
        ]
        return ChatPromptTemplate(input_variables=input_variables, messages=messages)

    @staticmethod
    def __preprocess(text: str):
        text = re.sub(r"\$(\d+)", r"$\1.0", text).lower()
        text = re.sub(r"\.(\d?)\.", r".", text)
        text = text.strip(".?")
        text = re.sub(
            r'\+|_|=|%|\^|\*|`|~|<|>|\||\?|\\|&|\$|\?|"', " ", text
        )  # Remove all punctuations
        text = re.sub(r"\{|\}|\[|\]", " ", text)
        text = re.sub(r" {2,}", " ", text)  # Remove extra spaces

        return text.strip()

    @classmethod
    def from_llm(
        cls,
        graph: DaiNeo4jGraph,
        llm: Optional[BaseLanguageModel] = None,
        cypher_llm: Optional[BaseLanguageModel] = None,
        qa_llm: Optional[BaseLanguageModel] = None,
        cypher_system_message: str = CYPHER_GENERATION_SYSTEM2,
        cypher_human_message: str = QUERY_INPUT_HUMAN,
        ner_modelpath: str = "Bert_NER",
        cypher_prompt_input_variables: Optional[List[str]] = None,
        exclude_types: List[str] = [],
        include_types: List[str] = [],
        validate_cypher: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        graph_schema = construct_schema(
            graph.get_structured_schema, include_types, exclude_types
        )

        if not cypher_llm and not llm:
            raise ValueError("Either `llm` or `cypher_llm` parameters must be provided")
        if not qa_llm and not llm:
            raise ValueError("Either `llm` or `qa_llm` parameters must be provided")
        if cypher_llm and qa_llm and llm:
            raise ValueError(
                "You can specify up to two of 'cypher_llm', 'qa_llm'"
                ", and 'llm', but not all three simultaneously."
            )

        qa_chain = LLMChain(llm=qa_llm or llm, prompt=CYPHER_QA_PROMPT)

        # cypher_prompt = cls.create_cypher_prompt(
        #     cypher_system_message, cypher_human_message, cypher_prompt_input_variables
        # )
        if cypher_prompt_input_variables is None:
            cypher_prompt_input_variables = ["schema", "question"]
        cypher_prompt = PromptTemplate(
            template=cypher_system_message + "\n" + cypher_human_message,
            input_variables=cypher_prompt_input_variables,
        )
        cypher_generation_chain = LLMChain(llm=cypher_llm or llm, prompt=cypher_prompt)

        cypher_query_corrector = None
        if validate_cypher:
            corrector_schema = [
                Schema(el["start"], el["type"], el["end"])
                for el in graph.structured_schema.get("relationships")
            ]
            cypher_query_corrector = CypherQueryCorrector(corrector_schema)

        return cls(
            graph=graph,
            graph_schema=graph_schema,
            qa_chain=qa_chain,
            cypher_generation_chain=cypher_generation_chain,
            cypher_query_corrector=cypher_query_corrector,
            entity_extractor=EntityExtractor2(
                llm=cypher_llm or llm, ner_modelpath=ner_modelpath
            ),  #
            verbose=verbose,
            **kwargs,
        )

    def __generate_heading(self, question: str):
        """
        Given an input question, generate a heading to precede the tabular result in rows
        """
        sent_starts = re.compile(
            "|".join(
                [
                    "list",
                    "get",
                    "describe",
                    "give",
                    "which",
                    "what",
                    "list",
                    "get",
                    "who",
                    "when",
                    "where",
                ]
            )
        )
        heading = sent_starts.sub("", question.lower()).strip(".").strip(" ").strip("?")
        if heading.startswith("is") or heading.startswith("are"):
            heading = "Following " + heading + ":"
        elif heading.startswith("how many"):
            heading = re.sub("how many", "The number of", heading) + " is:"
        elif " is " in heading:
            heading += " is:"
        else:
            heading += " are:"
        heading = heading[0].upper() + heading[1:]
        return heading

    def __validate_cypher(self, q: str) -> str:
        if "COUNT" in q.upper():
            if not re.search("COUNT.*\(.*DISTINCT", q, re.I):
                q = re.sub(r"COUNT\((.+)\)", r"COUNT(DISTINCT(\1))", q, flags=re.I)
        elif not re.search(r"RETURN DISTINCT ", q, flags=re.IGNORECASE | re.DOTALL):
            q = re.sub(
                r"RETURN (.+)",
                r"RETURN DISTINCT \1",
                q,
                flags=re.IGNORECASE | re.DOTALL,
            )
        q = re.sub(r"{\w+: (\w+)}", r"{number: \1}", q) # Only the `number` property should be used to directly identify the entities: all other properties are replaced
        m = re.search(r"RETURN (.+)", q, flags=re.I)
        # print(m)
        if m is not None:
            otext = m.groups()[0]
            text = re.sub("DISTINCT", "", otext, flags=re.I).strip(" ;")
            ret = set([t.strip() for t in text.split(",")])
            # print(ret)
            if (
                not any([a in q.lower() for a in ["sum", "avg", "count"]])
                and len(ret) < 2
            ):
                if "Customer" in q and "c" in q:
                    ret.add("c.number AS customer_number")
                    ret.add("c.name AS customer_name")
                if "Vendor" in q and "v" in q:
                    ret.add("v.number AS vendor_number")
                    ret.add("v.name AS vendor_name")
                if "Item" in q and "i" in q:
                    ret.add("i.number AS item_number")
                    ret.add("i.name AS item_name")
                if "SalesRep" in q and "srep" in q:
                    ret.add("srep.number AS sales_rep_number")
                    ret.add("srep.name AS sales_rep_name")
                if "SalesRegion" in q and "sreg" in q:
                    ret.add("sreg.number AS sales_reg_number")
                    ret.add("sreg.description AS sales_region_description")

                m2 = re.search(r"WHERE (.+)", q, flags=re.I)
                if m2 is not None:
                    where_clause = m2.groups()[0]
                    m3 = re.findall(r"[a-z]\.\w+", where_clause)
                    scls_to_fullcls = {
                        "v": "vendor",
                        "c": "customer",
                        "i": "item",
                        "srep": "sales_rep",
                        "sreg": "sales_region",
                        "r": "record",
                    }
                    for ms in m3:
                        cl, prop = ms.split(".")
                        # print(cl, prop)
                        if cl in scls_to_fullcls:
                            ret.add(f"{ms} AS {scls_to_fullcls[cl]}_{prop}")

                ret = ", ".join(sorted(list(ret), key=lambda a: a.lower()[0]))
                if " DISTINCT " in otext.upper():
                    ret = f"DISTINCT {ret}"
                if otext.endswith(";"):
                    ret = f"{ret};"
                # print(ret)
                nq = re.sub(r"RETURN (.+)", f"RETURN {ret}", q)
                return nq
        return q

    @staticmethod
    def apply_filters(cypher: str, params: Dict[str, str], filters: Dict[str, str]) -> Tuple[str, Dict[str, str]]:
        
        # TODO: Find variable names from the cypher instead of using v, c, r, srep, sreg by default
        
        def _apply(cypher, ncl, cond_adds):
            clines = cypher.split('\n')
            x = len(clines)
            cont = True
            i = 0
            while cont:
                cl = clines[i]
                if cl.upper().startswith('MATCH') and not clines[i+1].upper().startswith('MATCH'): #'MATCH' in cl:
                    # Get the last match statement and add the new line after that that last MATCH statement
                    # x = i
                    if len(ncl) > 0:
                        clines = clines[:i+1] + [ncl] + clines[i+1:]
                        ncl = ""
                        i += 1
                        # x += 1
                    if not re.search('WHERE', cypher, flags=re.IGNORECASE | re.MULTILINE):
                        # Add the WHERE clause after the last MATCH clause if the cypher doesn't already contain the WHERE clause
                        clines = clines[:i+1] + [f"WHERE {cond_adds}"] + clines[i+1:]
                        cond_adds = ""
                        i += 1
                elif cl.upper().startswith('WHERE') and len(cond_adds) > 0: #'WHERE' in cl:
                    # clines[i] = cl + f" AND c.chain =~ '{chain}'"
                    clines[i] = cl + f" AND ({cond_adds})"
                # elif not re.search('WHERE', cypher, flags=re.IGNORECASE | re.MULTILINE) and clines[i].startswith('RETURN'): #in cypher
                #     clines = clines[:i] + [f"WHERE {cond_adds}"] + clines[i:]
                #     x += 1
                #     i += 1
                cypher = '\n'.join(clines)
                i += 1
                x = len(clines)
                if i >= x:
                    cont = False
            # clines = clines[:x] + [ncl] + clines[x+1:]
            cypher = '\n'.join(clines)
            return cypher
        
        if 'brand' in filters and len(filters['brand']) > 0:
            brands = [br.strip().lower().split(' ') for br in filters['brand'].split(',')]
            brands = ['.*'.join(br) for br in brands]
            brands = [br.replace("'",".*") for br in brands]
            brands = [f"(?i).*{br}.*" for br in brands]
            cond_adds = ' OR '.join([f"i.brand_name =~ '{brand}'" for brand in brands])
            # print(cond_adds)
            # params['brand'] = brand
            ncl = ""
            if not 'i:Item' in cypher:
                if 'r:Record' in cypher:
                    ncl = "MATCH (r)-[:RECORDED_ITEM]-(i:Item)"
                elif 'v:Vendor' in cypher:
                    ncl = "MATCH (v)-[:SOLD_ITEM]-(i:Item)"
                elif 'c:Customer' in cypher:
                    ncl = "MATCH (c)-[:BOUGHT_ITEM]-(i:Item)"
                else:
                    ncl = "MATCH (i:Item)"
            cypher = _apply(cypher, ncl, cond_adds)
            
            #print(cypher)
        if 'chain' in filters and len(filters['chain']) > 0:
            chains = []
            for chain in filters['chain'].split(','):
                chain = filters['chain'].lower().split(' ')
                chain = '.*'.join(chain)
                chain = chain.replace("'",".*")
                chains.append(f"(?i).*{chain}.*")
            cond_adds = ' OR '.join([f"c.chain =~ '{chain}'" for chain in chains])
            # print(chain)
            ncl = ""
            if not 'c:Customer' in cypher:
                if 'r:Record' in cypher:
                    ncl = "MATCH (r)-[:RECORDED_CUSTOMER]-(c:Customer)"
                elif 'v:Vendor' in cypher:
                    ncl = "MATCH (v)-[:SELLER_CUSTOMER]-(c:Customer)"
                elif 'i:Item' in cypher:
                    ncl = "MATCH (c:Customer)-[:BOUGHT_ITEM]-(i)"
                else:
                    ncl = "MATCH (c:Customer)"
            
            cypher = _apply(cypher, ncl, cond_adds)
        
        if 'year' in filters and len(filters['year']) > 0:
            years = [y.strip().lower() for y in filters['year'].split(',') if is_number(y.strip())]
            years = ','.join(years)
            cond_adds = f"r.record_date.year IN [{years}]"
            # cond_adds = ' OR '.join([f"r.record_date.year = {year}" for year in years])
            
            ncl = ""
            if not "r:Record" in cypher:
                if 'v:Vendor' in cypher:
                    ncl = "MATCH (r:Record)-[:RECORDED_VENDOR]-(v)"
                elif 'c:Customer' in cypher:
                    ncl = "MATCH (r:Record)-[:RECORDED_CUSTOMER]-(c)"
                elif 'i:Item' in cypher:
                    ncl = "MATCH (r:Record)-[:RECORDED_ITEM]-(i)"
                elif 'srep:SalesRep' in cypher:
                    ncl = "MATCH (r:Record)-[:RECORDED_SALES_REP]-(srep)"
                elif 'sreg:SalesRegion' in cypher:
                    ncl = "MATCH (r:Record)-[:RECORDED_SALES_REGION]-(sreg)"
                else:
                    ncl = "MATCH (r:Record)"
            
            cypher = _apply(cypher, ncl, cond_adds)
        
        if 'month' in filters and len(filters['month']) > 0:
            months = []
            sh_months = ['','jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
            lg_months = [
                '',
                'january',
                'february',
                'march',
                'april',
                'may',
                'june',
                'july',
                'august',
                'september',
                'october',
                'november',
                'december'
            ]
            for month in filters['month'].split(','):
                month = month.strip()
                if month in ['1','2','3','4','5','6','7','9','10','11','12']:
                    months.append(month)
                elif month.lower() in sh_months:
                    months.append(str(sh_months.index(month.lower())))
                elif month.lower() in lg_months:
                    months.append(str(lg_months.index(month.lower())))
                else:
                    print(month)
            months = ','.join(months)
            cond_adds = f"r.record_date.month IN [{months}]"

            ncl = ""
            if not "r:Record" in cypher:
                if 'v:Vendor' in cypher:
                    ncl = "MATCH (r:Record)-[:RECORDED_VENDOR]-(v)"
                elif 'c:Customer' in cypher:
                    ncl = "MATCH (r:Record)-[:RECORDED_CUSTOMER]-(c)"
                elif 'i:Item' in cypher:
                    ncl = "MATCH (r:Record)-[:RECORDED_ITEM]-(i)"
                elif 'srep:SalesRep' in cypher:
                    ncl = "MATCH (r:Record)-[:RECORDED_SALES_REP]-(srep)"
                elif 'sreg:SalesRegion' in cypher:
                    ncl = "MATCH (r:Record)-[:RECORDED_SALES_REGION]-(sreg)"
                else:
                    ncl = "MATCH (r:Record)"
            
            cypher = _apply(cypher, ncl, cond_adds)

        return cypher, params

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        question = self.__preprocess(inputs[self.input_key])
        
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()

        params = None
        if self.entity_extractor is not None:
            entities = self.entity_extractor.extract_entities(question)
            _run_manager.on_text("Extracted Entities:", end="\n", verbose=self.verbose)
            _run_manager.on_text(
                f"{entities}", color="blue", end="\n", verbose=self.verbose
            )
            params = {}
            prop_counts = {
                "Vendor": 0,
                "Customer": 0,
                "Item": 0,
                "SalesRep": 0,
                "SalesRegion": 0,
            }
            pnames = {
                "Vendor": "vnos",
                "Customer": "cnos",
                "Item": "inos",
                "SalesRep": "srepnos",
                "SalesRegion": "sregnos",
            }
            """
            ent_pn = {}
            for ent, prop_id in zip(entities, self.graph.search_index_lazy(
                entities, top_k=10, max_distance=2.0, max_difference=5.0
            )):
                prop, ids = prop_id
                if len(ids) == 0:
                    continue
                c = prop_counts[prop]
                prop_counts[prop] = c + 1
                pname = pnames[prop]
                
                # ent_pn.append(f"{ent['text']} -> {pname}")
                # print(prop, c)
                if c == 0:
                    params[pname] = ids
                    ent_pn[ent['text']] = pname
                elif c == 1:
                    params[f"{pname}1"] = params.pop(pname)
                    params[f"{pname}2"] = ids
                    for e, pn in ent_pn.items():
                        if pn == pname:
                            ent_pn[e] = f"{pname}1"
                    ent_pn[ent['text']] = f"{pname}2"
                else:
                    params[f"{pname}{c+1}"] = ids
                    ent_pn[ent['text']] = f"{pname}{c+1}"
            """
            ent_pn = {pn: "" for pn in pnames.values()}
            for ent, prop_id in zip(
                entities,
                self.graph.search_index_lazy(
                    entities, top_k=10, max_distance=2.0, max_difference=5.0
                ),
            ):
                prop, ids = prop_id
                if len(ids) == 0:
                    continue
                pname = pnames[prop]
                ids = params.get(pname, []) + ids
                params[pname] = ids
                ent_pn[pname] += f",{ent['text']}"
            ent_pn = {v.strip(", "): k for k, v in ent_pn.items() if v != ""}

            # print(ent_pn)
            _run_manager.on_text(
                "Identified Entities from Index:", end="\n", verbose=self.verbose
            )
            _run_manager.on_text(
                f"{params}", color="yellow", end="\n", verbose=self.verbose
            )

        generated_cypher = self.cypher_generation_chain.invoke(
            {
                "question": question,
                "schema": self.graph_schema,
                "params": str(ent_pn).strip("{}"),
            },
            callbacks=callbacks,
        )[self.cypher_generation_chain.output_key]

        # Extract Cypher code if it is wrapped in backticks
        generated_cypher = extract_cypher(generated_cypher).strip('\n ')

        # Correct Cypher query if enabled
        if self.cypher_query_corrector:
            generated_cypher = self.cypher_query_corrector(generated_cypher)

        m = re.search(r"RETURN (.+)", generated_cypher, flags=re.I)
        rets = []
        if m is not None:
            otext = m.groups()[0].strip(";")
            rets = [r.strip() for r in otext.split(",")]
        else:
            return {
                self.output_heading_key: "Sorry! There was a problem understanding your query. Maybe try rewording your query...",
                self.output_rows_key: "",
                self.output_suggs_key: [],
                self.output_text_result_key: "",
            }
        generated_cypher = self.__validate_cypher(generated_cypher)
        if (
            any([a in generated_cypher.lower() for a in ["sum", "avg", "count"]])
            and len(rets) < 2
        ):
            for k, v in params.items():
                try:
                    params[k] = v[0]
                except:
                    pass
        if "top" in question or "bottom" in question:
            if not "LIMIT" in generated_cypher.upper():
                top = re.findall(r"[top|bottom] (\d+)", question, flags=re.I)
                if len(top) > 0:
                    top = top[0]
                else:
                    top = 10
                generated_cypher = f"{generated_cypher.strip(';')} LIMIT {top};"

        elif "LIMIT" in generated_cypher.upper():
            generated_cypher = re.sub(
                r"LIMIT \d+", "LIMIT 50", generated_cypher, flags=re.I
            )
        # print(generated_cypher)
        filters = inputs.get(self.filters_key, None)
        if filters is not None:
            generated_cypher, params = self.apply_filters(generated_cypher, params, filters)

        _run_manager.on_text("Generated Cypher:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            generated_cypher, color="blue", end="\n", verbose=self.verbose
        )
        qparams = re.findall(r"\$([a-z0-9]+) ", generated_cypher, flags=re.I)
        # print(qparams, params.keys())
        if len(set(qparams) - set(params.keys())) > 0:
            # If there are extra params in generated query than in the input params, return an error message
            return {
                self.output_heading_key: "Sorry! There was a problem in answering your question. Maybe try rephrasing your query...",
                self.output_rows_key: "",
                self.output_suggs_key: [],
                self.output_text_result_key: "",
            }
        else:
            eparams = set(params.keys()) - set(qparams)
            if len(eparams) > 0:
                for ep in eparams:
                    params.pop(ep)

        query_cls = "table"  # self.query_classifier.classify(question)
        if query_cls == "table":
            try:
                answer = self.graph.query(
                    generated_cypher, params=params, return_dframe=True
                )
            except ValueError as ve:
                print(ve)
                return {
                    self.output_heading_key: "Sorry! There was a problem understanding your query. Please try rewording your query...",
                    self.output_rows_key: "",
                    self.output_suggs_key: [],
                    self.output_text_result_key: "",
                }
            if len(answer) == 0:
                return {
                    self.output_heading_key: "Sorry! No answer to query found in the Knowledge Graph! Please try another query.",
                    self.output_rows_key: "",
                    self.output_suggs_key: [],
                    self.output_text_result_key: "",
                }

            # formatters = {}
            for col in answer.columns:
                if any(
                    [
                        x in col
                        for x in ["rep", "region", "number", "name", "description"]
                    ]
                ):
                    continue
                if any([x in col for x in ["cost", "price", "sales", "srp", "retail"]]):
                    # formatters[col] = partial(format_float,form="${:,.2f}")
                    answer[col] = answer[col].apply(
                        partial(format_float, form="${:,.2f}")
                    )
                elif any([x in col for x in ["weight"]]):
                    # formatters[col] = partial(format_float,form="{:,.2f} lb")
                    answer[col] = answer[col].apply(
                        partial(format_float, form="{:,.2f} lb")
                    )
                elif any([x in col for x in ["cases"]]):
                    # formatters[col] = partial(format_float,form="{:,}")#.format(a) if not isinstance(a,str) else a
                    answer[col] = answer[col].apply(
                        partial(format_float, form="{:,.0f}")
                    )
                elif any([x in col for x in ["margin"]]):
                    answer[col] = answer[col].apply(
                        partial(format_float, form="{:,.2f}%")
                    )
            header = []
            for col in answer.columns:
                if not "." in col:
                    header.append(convertToTitleCase(col, all_upper=["upc", "srp"]))
                else:
                    header.append(col)
            if len(header) != len(answer.columns):
                header = True
            heading = self.__generate_heading(question)
            answer = answer.to_csv(
                sep="|",
                index=False,
                na_rep="Not Available",
                float_format=lambda a: "{:,.2f}".format(a),
                header=header,
                quoting=csv.QUOTE_NONE,
                escapechar="\\",
            )
            return {
                self.output_heading_key: heading,
                self.output_rows_key: answer.strip("\n "),
                self.output_suggs_key: [],
                self.output_text_result_key: "",
            }
        elif query_cls == "text":
            try:
                context = self.graph.query(
                    generated_cypher,
                    params=params,
                    return_dframe=False,
                    return_dict=True,
                )
            except ValueError as ve:
                print(ve)
                return {
                    self.output_heading_key: "Sorry! There was a problem in answering your question. Please try rephrasing your query...",
                    self.output_rows_key: "",
                    self.output_suggs_key: [],
                    self.output_text_result_key: "",
                }
            except Exception as e:
                print(e)
                return {
                    self.output_heading_key: "Sorry! There was a problem in answering your question. Please try rephrasing your query...",
                    self.output_rows_key: "",
                    self.output_suggs_key: [],
                    self.output_text_result_key: "",
                }
            _run_manager.on_text(
                f"Received context:\n{context}",
                color="green",
                end="\n",
                verbose=self.verbose,
            )
            context = f"{context[:self.top_k]}"
            answer = self.qa_chain.invoke(
                {"question": question, "context": context},
                return_only_outputs=True,
                callbacks=callbacks,
            )[self.qa_chain.output_key]
            return {
                self.output_heading_key: "",
                self.output_rows_key: "",
                self.output_suggs_key: [],
                self.output_text_result_key: answer,
            }

    def answer_question(self, query: str) -> str:
        filters = None
        if re.search("FILTERS:", query, flags=re.I):
            query_filters = re.split(r"FILTERS:", query)
            query = query_filters[0].strip()
            filters = json.loads(query_filters[1].strip())

        # query = inputs.pop('query')
        # filters = inputs.pop('filters', None)
        return json.dumps(self.invoke({self.input_key: query, self.filters_key: filters}, return_only_outputs=True))
        # return json.dumps(self.invoke(inputs, return_only_outputs=True))

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        question = self.__preprocess(inputs[self.input_key])

        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()

        params = None
        ent_pn = {}
        if self.entity_extractor is not None:
            entities = await self.entity_extractor.async_extract_entities(question)
            await _run_manager.on_text(
                "Extracted Entities:", end="\n", verbose=self.verbose
            )
            await _run_manager.on_text(
                f"{entities}", color="blue", end="\n", verbose=self.verbose
            )
            params = {}
            pnames = {
                "Vendor": "vnos",
                "Customer": "cnos",
                "Item": "inos",
                "SalesRep": "srepnos",
                "SalesRegion": "sregnos",
            }
            ent_pn = {pn: "" for pn in pnames.values()}
            for ent, prop_id in zip(
                entities,
                self.graph.search_index_lazy(
                    entities, top_k=10, max_distance=2.0, max_difference=5.0
                ),
            ):
                prop, ids = prop_id
                if len(ids) == 0:
                    continue
                pname = pnames[prop]
                ids = params.get(pname, []) + ids
                params[pname] = ids
                ent_pn[pname] += f",{ent['text']}"
            ent_pn = {v.strip(", "): k for k, v in ent_pn.items() if v != ""}

            # print(ent_pn)
            await _run_manager.on_text(
                "Identified Entities from Index:", end="\n", verbose=self.verbose
            )
            await _run_manager.on_text(
                f"{params}", color="yellow", end="\n", verbose=self.verbose
            )

        generated_cypher = await self.cypher_generation_chain.ainvoke(
            {
                "question": question,
                "schema": self.graph_schema,
                "params": str(ent_pn).strip("{}"),
            },
            callbacks=callbacks,
            return_only_outputs=True,
        )[self.cypher_generation_chain._run_output_key]

        # Extract Cypher code if it is wrapped in backticks
        generated_cypher = extract_cypher(generated_cypher).strip('\n')

        # Correct Cypher query if enabled
        if self.cypher_query_corrector:
            generated_cypher = self.cypher_query_corrector(generated_cypher)

        m = re.search(r"RETURN (.+)", generated_cypher, flags=re.I)
        rets = []
        if m is not None:
            otext = m.groups()[0].strip(";")
            rets = [r.strip() for r in otext.split(",")]
        else:
            return {
                self.output_heading_key: "Sorry! There was a problem understanding your query. Maybe try rewording your query...",
                self.output_rows_key: "",
                self.output_suggs_key: [],
                self.output_text_result_key: "",
            }
        generated_cypher = self.__validate_cypher(generated_cypher).strip('\n')
        if (
            any([a in generated_cypher.lower() for a in ["sum", "avg", "count"]])
            and len(rets) < 2
        ):
            for k, v in params.items():
                try:
                    params[k] = v[0]
                except:
                    pass
        if "top" in question or "bottom" in question:
            if not "LIMIT" in generated_cypher.upper():
                top = re.findall(r"[top|bottom] (\d+)", question, flags=re.I)
                if len(top) > 0:
                    top = top[0]
                else:
                    top = 10
                generated_cypher = f"{generated_cypher.strip(';')} LIMIT {top};"

        elif "LIMIT" in generated_cypher.upper():
            generated_cypher = re.sub(
                r"LIMIT \d+", "LIMIT 50", generated_cypher, flags=re.I
            )
        
        filters = inputs.get(self.filters_key, None)
        if filters is not None:
            generated_cypher, params = self.apply_filters(generated_cypher, params, filters)

        await _run_manager.on_text("Generated Cypher:", end="\n", verbose=self.verbose)
        await _run_manager.on_text(
            generated_cypher, color="blue", end="\n", verbose=self.verbose
        )
        qparams = re.findall(r"\$([a-z0-9]+) ", generated_cypher, flags=re.I)
        # print(qparams, params.keys())
        if len(set(qparams) - set(params.keys())) > 0:
            # If there are extra params in generated query than in the input params, return an error message
            return {
                self.output_heading_key: "Sorry! There was a problem in answering your question. Maybe try rephrasing your query...",
                self.output_rows_key: "",
                self.output_suggs_key: [],
                self.output_text_result_key: "",
            }
        else:
            eparams = set(params.keys()) - set(qparams)
            if len(eparams) > 0:
                for ep in eparams:
                    params.pop(ep)

        query_cls = "table"  # self.query_classifier.classify(question)
        if query_cls == "table":
            try:
                answer = self.graph.query(
                    generated_cypher, params=params, return_dframe=True
                )
            except ValueError as ve:
                print(ve)
                return {
                    self.output_heading_key: "Sorry! There was a problem understanding your query. Please try another query...",
                    self.output_rows_key: "",
                    self.output_suggs_key: [],
                    self.output_text_result_key: "",
                }
            if len(answer) == 0:
                return {
                    self.output_heading_key: "Sorry! No answer to query found in the Knowledge Graph!",
                    self.output_rows_key: "",
                    self.output_suggs_key: [],
                    self.output_text_result_key: "",
                }

            # formatters = {}
            for col in answer.columns:
                if any(
                    [
                        x in col
                        for x in ["rep", "region", "number", "name", "description"]
                    ]
                ):
                    continue
                if any([x in col for x in ["cost", "price", "sales", "srp", "retail"]]):
                    # formatters[col] = partial(format_float,form="${:,.2f}")
                    answer[col] = answer[col].apply(
                        partial(format_float, form="${:,.2f}")
                    )
                elif any([x in col for x in ["weight"]]):
                    # formatters[col] = partial(format_float,form="{:,.2f} lb")
                    answer[col] = answer[col].apply(
                        partial(format_float, form="{:,.2f} lb")
                    )
                elif any([x in col for x in ["cases"]]):
                    # formatters[col] = partial(format_float,form="{:,}")#.format(a) if not isinstance(a,str) else a
                    answer[col] = answer[col].apply(
                        partial(format_float, form="{:,.0f}")
                    )
                elif any([x in col for x in ["margin"]]):
                    answer[col] = answer[col].apply(
                        partial(format_float, form="{:,.2f}%")
                    )
            header = []
            for col in answer.columns:
                if not "." in col:
                    header.append(convertToTitleCase(col, all_upper=["upc", "srp"]))
                else:
                    header.append(col)
            if len(header) != len(answer.columns):
                header = True
            heading = self.__generate_heading(question)
            answer = answer.to_csv(
                sep="|",
                index=False,
                na_rep="Not Available",
                float_format=lambda a: "{:,.2f}".format(a),
                header=header,
                quoting=csv.QUOTE_NONE,
                escapechar="\\",
            )
            return {
                self.output_heading_key: heading,
                self.output_rows_key: answer.strip("\n "),
                self.output_suggs_key: [],
                self.output_text_result_key: "",
            }
        elif query_cls == "text":
            try:
                context = self.graph.query(
                    generated_cypher,
                    params=params,
                    return_dframe=False,
                    return_dict=True,
                )
            except ValueError as ve:
                print(ve)
                return {
                    self.output_heading_key: "Sorry! There was a problem in answering your question. Please try rephrasing your query...",
                    self.output_rows_key: "",
                    self.output_suggs_key: [],
                    self.output_text_result_key: "",
                }
            except Exception as e:
                print(e)
                return {
                    self.output_heading_key: "Sorry! There was a problem in answering your question. Maybe try rephrasing your query...",
                    self.output_rows_key: "",
                    self.output_suggs_key: [],
                    self.output_text_result_key: "",
                }
            await _run_manager.on_text(
                f"Received context:\n{context}",
                color="green",
                end="\n",
                verbose=self.verbose,
            )
            context = f"{context[:self.top_k]}"
            answer = await self.qa_chain.ainvoke(
                {"question": question, "context": context},
                return_only_outputs=True,
                callbacks=callbacks,
            )[self.qa_chain.output_key]
            return {
                self.output_heading_key: "",
                self.output_rows_key: "",
                self.output_suggs_key: [],
                self.output_text_result_key: answer,
            }

    async def async_answer_question(self, inputs: Dict[str, str]) -> str:
        filters = None
        if re.search("FILTERS:", query, flags=re.I):
            query_filters = re.split(r"FILTERS:", query)
            query = query_filters[0].strip()
            filters = json.loads(query_filters[1].strip())

        # query = inputs.pop('query')
        # filters = inputs.pop('filters', None)
        return await json.dumps(self.ainvoke({self.input_key: query, self.filters_key: filters}, return_only_outputs=True))
        # return json.dumps(await self.ainvoke(inputs, return_only_outputs=True))

    def _get_run_manager(self, query, is_async=False):
        # from langchain.callbacks.manager import CallbackManager
        from langchain.load.dump import dumpd

        if not is_async:
            callback_manager = CallbackManager.configure(
                None,
                self.callbacks,
                self.verbose,
                None,
                self.tags,
                None,
                self.metadata,
            )
        else:
            callback_manager = AsyncCallbackManager.configure(
                None,
                self.callbacks,
                self.verbose,
                None,
                self.tags,
                None,
                self.metadata,
            )
        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            {self.input_key: query},
            name=None,
        )
        return run_manager

    def search(self, query: str) -> str:
        
        filters = None
        if re.search("FILTERS:", query, flags=re.I):
            query_filters = re.split(r"FILTERS:", query)
            query = query_filters[0].strip()
            filters = json.loads(query_filters[1].strip())
        # filters = inputs.pop(self.filters_key, None)
        # query = inputs[self.input_key]
        question = self.__preprocess(query)

        _run_manager = self._get_run_manager(query)
        callbacks = _run_manager.get_child()

        params = None
        ent_pn = ""
        if self.entity_extractor is not None:
            entities = self.entity_extractor.extract_entities(question)
            _run_manager.on_text("Extracted Entities:", end="\n", verbose=self.verbose)
            _run_manager.on_text(
                f"{entities}", color="blue", end="\n", verbose=self.verbose
            )
            params = {}
            pnames = {
                "Vendor": "vnos",
                "Customer": "cnos",
                "Item": "inos",
                "SalesRep": "srepnos",
                "SalesRegion": "sregnos",
            }
            ent_pn = {pn: "" for pn in pnames.values()}
            for ent, prop_id in zip(
                entities,
                self.graph.search_index_lazy(
                    entities, top_k=10, max_distance=2.0, max_difference=5.0
                ),
            ):
                prop, ids = prop_id
                if len(ids) == 0:
                    continue
                pname = pnames[prop]
                ids = params.get(pname, []) + ids
                params[pname] = ids
                ent_pn[pname] += f",{ent['text']}"
            ent_pn = {v.strip(", "): k for k, v in ent_pn.items() if v != ""}

            # print(ent_pn)
            _run_manager.on_text(
                "Identified Entities from Index:", end="\n", verbose=self.verbose
            )
            _run_manager.on_text(
                f"{params}", color="yellow", end="\n", verbose=self.verbose
            )

        generated_cypher = self.cypher_generation_chain.invoke(
            {
                "question": question,
                "schema": self.graph_schema,
                "params": str(ent_pn).strip("{}"),
            },
            callbacks=callbacks,
            return_only_outputs=True,
        )[self.cypher_generation_chain.output_key]

        # Extract Cypher code if it is wrapped in backticks
        generated_cypher = extract_cypher(generated_cypher)

        # Correct Cypher query if enabled
        if self.cypher_query_corrector:
            generated_cypher = self.cypher_query_corrector(generated_cypher)

        m = re.search(r"RETURN (.+)", generated_cypher, flags=re.I)
        rets = []
        if m is not None:
            otext = m.groups()[0].strip(";")
            rets = [r.strip() for r in otext.split(",")]
        else:
            return {
                self.output_heading_key: "Sorry! There was a problem understanding your query. Maybe try rephrasing your query...",
                self.output_rows_key: "",
                self.output_suggs_key: [],
                self.output_text_result_key: "",
            }
        generated_cypher = self.__validate_cypher(generated_cypher)
        if (
            any([a in generated_cypher.lower() for a in ["sum", "avg", "count"]])
            and len(rets) < 2
        ):
            for k, v in params.items():
                try:
                    params[k] = v[0]
                except:
                    pass
        if "top" in question or "bottom" in question:
            if not "LIMIT" in generated_cypher.upper():
                top = re.findall(r"[top|bottom] (\d+)", question, flags=re.I)
                if len(top) > 0:
                    top = top[0]
                else:
                    top = 10
                generated_cypher = f"{generated_cypher.strip(' ;')} LIMIT {top};"

        elif "LIMIT" in generated_cypher.upper():
            generated_cypher = re.sub(
                r"LIMIT \d+", "LIMIT 50", generated_cypher, flags=re.I
            )

        if filters is not None:
            generated_cypher, params = self.apply_filters(generated_cypher, params, filters)

        _run_manager.on_text("Generated Cypher:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            generated_cypher, color="blue", end="\n", verbose=self.verbose
        )
        qparams = re.findall(r"\$([a-z0-9]+) ", generated_cypher, flags=re.I)
        # print(qparams, params.keys())
        if len(set(qparams) - set(params.keys())) > 0:
            # If there are extra params in generated query than in the input params, return an error message
            return {
                self.output_heading_key: "Sorry! There was a problem in answering your question. Maybe try rephrasing your query...",
                self.output_rows_key: "",
                self.output_suggs_key: [],
                self.output_text_result_key: "",
            }
        else:
            eparams = set(params.keys()) - set(qparams)
            if len(eparams) > 0:
                for ep in eparams:
                    params.pop(ep)
        try:
            context = self.graph.query(
                generated_cypher, params=params, return_dframe=False, return_dict=True
            )
        except ValueError as ve:
            print(ve)
            return {
                self.output_heading_key: "Sorry! There was a problem in answering your question. Please try rephrasing your query...",
                self.output_rows_key: "",
                self.output_suggs_key: [],
                self.output_text_result_key: "",
            }
        except Exception as e:
            print(e)
            return {
                self.output_heading_key: "Sorry! There was a problem in answering your question. Please try rephrasing your query...",
                self.output_rows_key: "",
                self.output_suggs_key: [],
                self.output_text_result_key: "",
            }
        # _run_manager.on_text(
        #     f"Received context:\n{context}",
        #     color="green",
        #     end="\n",
        #     verbose=self.verbose,
        # )
        context = f"{context[:self.top_k]}"
        return context

        answer = self.qa_chain.invoke(
            {"question": question, "context": context},
            return_only_outputs=True,
            callbacks=callbacks,
        )[self.qa_chain.output_key]
        return {
            self.output_heading_key: "",
            self.output_rows_key: "",
            self.output_suggs_key: [],
            self.output_text_result_key: answer,
        }

    async def asearch(self, query: str) -> str:
        
        filters = None
        if re.search("FILTERS:", query, flags=re.I):
            query_filters = re.split(r"FILTERS:", query)
            query = query_filters[0].strip()
            filters = json.loads(query_filters[1].strip())
        # filters = inputs.pop(self.filters_key, None)
        # query = inputs[self.input_key]
        question = self.__preprocess(query)

        _run_manager = self._get_run_manager(query, is_async=True)
        callbacks = _run_manager.get_child()

        params = None
        ent_pn = ""
        if self.entity_extractor is not None:
            entities = await self.entity_extractor.async_extract_entities(question)
            await _run_manager.on_text(
                "Extracted Entities:", end="\n", verbose=self.verbose
            )
            await _run_manager.on_text(
                f"{entities}", color="blue", end="\n", verbose=self.verbose
            )
            params = {}
            pnames = {
                "Vendor": "vnos",
                "Customer": "cnos",
                "Item": "inos",
                "SalesRep": "srepnos",
                "SalesRegion": "sregnos",
            }
            ent_pn = {pn: "" for pn in pnames.values()}
            for ent, prop_id in zip(
                entities,
                self.graph.search_index_lazy(
                    entities, top_k=10, max_distance=2.0, max_difference=5.0
                ),
            ):
                prop, ids = prop_id
                if len(ids) == 0:
                    continue
                pname = pnames[prop]
                ids = params.get(pname, []) + ids
                params[pname] = ids
                ent_pn[pname] += f",{ent['text']}"
            ent_pn = {v.strip(", "): k for k, v in ent_pn.items() if v != ""}

            # print(ent_pn)
            await _run_manager.on_text(
                "Identified Entities from Index:", end="\n", verbose=self.verbose
            )
            await _run_manager.on_text(
                f"{params}", color="yellow", end="\n", verbose=self.verbose
            )

        generated_cypher = await self.cypher_generation_chain.ainvoke(
            {
                "question": question,
                "schema": self.graph_schema,
                "params": str(ent_pn).strip("{}"),
            },
            callbacks=callbacks,
            return_only_outputs=True,
        )[self.cypher_generation_chain.output_key]

        # Extract Cypher code if it is wrapped in backticks
        generated_cypher = extract_cypher(generated_cypher)

        # Correct Cypher query if enabled
        if self.cypher_query_corrector:
            generated_cypher = self.cypher_query_corrector(generated_cypher)

        m = re.search(r"RETURN (.+)", generated_cypher, flags=re.I)
        rets = []
        if m is not None:
            otext = m.groups()[0].strip(";")
            rets = [r.strip() for r in otext.split(",")]
        else:
            return {
                self.output_heading_key: "Sorry! There was a problem understanding your query. Maybe try rephrasing your query...",
                self.output_rows_key: "",
                self.output_suggs_key: [],
                self.output_text_result_key: "",
            }
        generated_cypher = self.__validate_cypher(generated_cypher)
        if (
            any([a in generated_cypher.lower() for a in ["sum", "avg", "count"]])
            and len(rets) < 2
        ):
            for k, v in params.items():
                try:
                    params[k] = v[0]
                except:
                    pass
        if "top" in question or "bottom" in question:
            if not "LIMIT" in generated_cypher.upper():
                top = re.findall(r"[top|bottom] (\d+)", question, flags=re.I)
                if len(top) > 0:
                    top = top[0]
                else:
                    top = 10
                generated_cypher = f"{generated_cypher.strip(' ;')} LIMIT {top};"

        elif "LIMIT" in generated_cypher.upper():
            generated_cypher = re.sub(
                r"LIMIT \d+", "LIMIT 50", generated_cypher, flags=re.I
            )

        if filters is not None:
            generated_cypher, params = self.apply_filters(generated_cypher, params, filters)

        await _run_manager.on_text("Generated Cypher:", end="\n", verbose=self.verbose)
        await _run_manager.on_text(
            generated_cypher, color="blue", end="\n", verbose=self.verbose
        )
        qparams = re.findall(r"\$([a-z0-9]+) ", generated_cypher, flags=re.I)
        # print(qparams, params.keys())
        if len(set(qparams) - set(params.keys())) > 0:
            # If there are extra params in generated query than in the input params, return an error message
            return {
                self.output_heading_key: "Sorry! There was a problem in answering your question. Maybe try rephrasing your query...",
                self.output_rows_key: "",
                self.output_suggs_key: [],
                self.output_text_result_key: "",
            }
        else:
            eparams = set(params.keys()) - set(qparams)
            if len(eparams) > 0:
                for ep in eparams:
                    params.pop(ep)
        try:
            context = self.graph.query(
                generated_cypher, params=params, return_dframe=False, return_dict=True
            )
        except ValueError as ve:
            print(ve)
            return {
                self.output_heading_key: "Sorry! There was a problem in answering your question. Please try rephrasing your query...",
                self.output_rows_key: "",
                self.output_suggs_key: [],
                self.output_text_result_key: "",
            }
        except Exception as e:
            print(e)
            return {
                self.output_heading_key: "Sorry! There was a problem in answering your question. Please try rephrasing your query...",
                self.output_rows_key: "",
                self.output_suggs_key: [],
                self.output_text_result_key: "",
            }
        # _run_manager.on_text(
        #     f"Received context:\n{context}",
        #     color="green",
        #     end="\n",
        #     verbose=self.verbose,
        # )
        context = f"{context[:self.top_k]}"
        return context

        answer = await self.qa_chain.acall(
            {"question": question, "context": context},
            return_only_outputs=True,
            callbacks=callbacks,
        )[self.qa_chain.output_key]
        return {
            self.output_heading_key: "",
            self.output_rows_key: "",
            self.output_suggs_key: [],
            self.output_text_result_key: answer,
        }
