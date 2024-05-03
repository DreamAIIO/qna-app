MAP_TEMPLATE = """The following is a set of documents
{docs}
Based on this list of docs, please identify the main themes 
Helpful Answer:"""

REDUCE_TEMPLATE = """The following is set of summaries:
{doc_summaries}
Take these and distill it into a final, consolidated summary of the main themes. 
Helpful Answer:"""

'''
Context: Lipari
Amish Wedding Southern Macaroni Salad
Size
5 lb
UPC
04964926138
Description: We've taken our Southern Style Potato Salad recipe and transformed it into macaroni salad. Our own
mustard dressing blend with elbow macaroni, peppers, onions, carrots, celery, pickle relish, and southern
seasonings.
Ingredients: Elbow Macaroni (Water, Pasta (Semolina, Niacin, Iron (Ferrous Sulfate), Thiamine Mononitrate,
Riboflavin, Folic Acid)), Salad Dressing (Soybean Oil, Water, High Fructose Corn Syrup, Distilled Vinegar, Modified
Corn Starch, Egg Yolk, Salt, Mustard Flour, Onion Powder, Calcium Disodium EDTA (to protect flavor), Natural
Flavors), Relish (Cucumbers, Sugar, Vinegar, Salt, Red Bell Pepper, Spices, Potassium Sorbate (Preservative),
Xanthan Gum, Yellow 5, Polysorbate 80), Sugar, Mustard (Distilled Vinegar, Water, Mustard Seed, Salt, Turmeric,
Paprika, Spice, Natural Flavor, Garlic Powder), Celery, Water, Red Peppers (Peppers, Water, Sugar, Salt, Citric Acid),
Onion, Green Peppers, Salt, Spice, Less than 1/10 of 1% Sodium Benzoate (Preservative).
CONTAINS: WHEAT, EGG
CONTAINS: Wheat, Eggs, Soybeans
Product Specifications
BrandManufactureName
Name
Kosher Certification:
Kosher Status:
Packaging
Case Dimension
TI/HI
Country of Origin
Amish Wedding
Herold’s Salads, Inc
NO
NO
5 lb tub
13 5/8 x 6 7/8 x 6 3/16
17/6
USA
Serving Size:
Shelf Life from date of
manufacture
Storage
Shelf Life (Ambient):
Net Weight
Gross Weight
4 oz
40 days
Refrigerated
0 days
5 lbs
10.85 lbs
Nutrition
Nutrion Facts:
1 serving per container
Serving Size 4oz (113g)
Calories 270 per serving
Amount per serving % Daily Value
Total Fat 14g 18%
Saturated Fat 2g 10%
Trans Fat 0g
Cholestrol 15mg 5%
Sodium 580mg 25%
Total Carbohydrate 31g 11%
Dietary Fiber 1g 5%
Total Sugars 10g
Includes 3g Added Sugars 7%
Protein 4g
Subject: A description of the product named Lipari Amish Wedding Southern Macaroni Salad
Keywords: Lipari Amish Wedding Southern Macaroni Salad, Product Specifications, Nutrition Facts, Amish Wedding
Summary: Lipari Amish Wedding Southern Macaroni Salad (UPC: 04964926138 and Size: 5lb).\nDescription: Southern Potato Salad transformed into macaroni salad with mustard dressing blend with elbow macaroni, pepperoni, peppers, onions, carrots, celery, pickle relish, and southern seasonings. Ingredients: Elbow Macaroni, Salad Dressing, Relish, Sugar, Mustard, Celery, Water, Red Peppers, Onion, Green Peppers, Salt, Spice, Less than 1/10 of 1% Sodium Benzoate, Wheat, Eggs, Soybeans.\nBrand: Amish Wedding, Manufactured by: Harold's Salads, Inc.\nNutrition Facts:\n1 serving per container,\nServing Size: 4oz (113g),\nCalories 270 per serving.\nTotal Fat 14g (18%),\nTotal Carbohydrate 31g (11%),\nCholestrol 15mg (5%),\nSodium 580mg (25%), Protein 4g.
'''


SUMMARY_PROMPT_TEMPLATE = """Summarise the following single page of text from a large document.
Make sure to follow the following steps when summarising the text:
1. Subject: Identify the subject i.e. what this text is about - e.g. is it a description or catalogue of a particular item, location etc. or a sales report or a sales flyer?
2. Keywords: Extract any keywords and major headings that help describe the text.
3. Summary: Generate a summary of 500-800 words of the text as a single paragraph. Always include table data.
Return the answer for each step as a part of JSON string with the following keys: 'Subject', 'Keywords', 'Summary'.

Context: {context}
Answer:"""


# QUERY: List all the items mentioned in the custom lipari file.
# =========
# Lipari Amish Wedding Southern Macaroni Salad (UPC: 04964926138 and Size: 5lb).\nDescription: Southern Potato Salad transformed into macaroni salad with mustard dressing blend with elbow macaroni, pepperoni, peppers, onions, carrots, celery, pickle relish, and southern seasonings. Ingredients: Elbow Macaroni, Salad Dressing, Relish, Sugar, Mustard, Celery, Water, Red Peppers, Onion, Green Peppers, Salt, Spice, Less than 1/10 of 1% Sodium Benzoate, Wheat, Eggs, Soybeans.\nBrand: Amish Wedding, Manufactured by: Harold's Salads, Inc.\nNutrition Facts:\n1 serving per container,\nServing Size: 4oz (113g),\nCalories 270 per serving.\nTotal Fat 14g (18%),\nTotal Carbohydrate 31g (11%),\nCholestrol 15mg (5%),\nSodium 580mg (25%), Protein 4g.
# Source: test_file.pdf: p. 1

# Lipari Amish Wedding Cranberry Orange Relish (UPC: 049646926107 and Size: 5lb).\nDescription: Fresh cranberries & oranges make this sweet and tangy treat. (Nov-Dec) Ingredients: Cranberries, Oranges, Sugar, Sodium Benzoate, Potassium Sorbate.\nBrand: Amish Wedding, Manufactured by: Harold's Salads, Inc.\nNutrition Facts:\n1 serving per container,\nServing Size: 4oz (113g),\nCalories 160 per serving.\nTotal Fat 0g,\nTotal Carbohydrate 41g (11%),\nCholestrol 0g,\nSodium 15mg (25%), Protein 1g.
# Source: test_file.pdf: p. 2

# Lipari Rotini Garden Medley Pasta (UPC: 04964926084 and Size: 5lb) are rotini pasta noodles mixed with a medley of carrots, red and green peppers, onions, broccoli florets, red beans, and baby corn in a tasty golden Italian dressing. Some major ingredients include tri-color rotini, italian Dressing, sugar, kidney Beans, corn syrup, onion, red & green Peppers, carrots and baby corn. It has the brand amish wedding and is manufactured by Harold's Salads, Inc. Its packaging is a 5 lb tub with case dimensions 13 5/8 x 6 7/8 x 6 3/16 and has an ambient shelf life of 40 days. For a single serving, its nutritional details are as follows: 160 calories, 4.5g total fat (5% DV), 0.5g saturated fat (3% DV), 0g trans fat, 0mg cholestrol, 250mg sodium (11% DV), 26g carbohydrate (10% DV), 4g dietary fiber (15% DV), 7g total sugars, 4g protein, 0mcg Vitamin D, 20mg Calcium (2% DV), 0.7mg Iron (4% DV) and 110mg Potassium (2% DV).
# Source: test_file.pdf: p. 3
# =========
# FINAL ANSWER: 1. Lipari Amish Wedding Southern Macaroni Salad,\n2. Lipari Amish Wedding Cranberry Orange Relish,\n3. Lipari Rotini Garden Medley Pasta\nSOURCES: test_file.pdf

#  with references ("SOURCES").

QA_PROMPT_REDUCE_TEMPLATE = """Given the following summaries of a page of a document and a query, create a final answer.
If the query asks for a summary, return a consolidated summary. If it asks for a list, return a numbered list. If it asks for a description, give a one-line description.
Your answer should be detailed and easy to read with good use of whitespace.

QUERY: {question}
=========
{doc_summaries}
=========
FINAL ANSWER:
"""

QA_PROMPT_REDUCE_TEMPLATE2 = """Given the following summaries of a page of a document and a query, create a final answer with references ("SOURCES").
If the query asks for a summary, return a consolidated summary. If it asks for a list, return a numbered list. If it asks for a description, give a one-line description. If you

EXAMPLES
====================================
1)
QUERY: List all the items mentioned in the custom lipari file.
=========
Lipari Amish Wedding Southern Macaroni Salad (UPC: 04964926138 and Size: 5lb).\nDescription: Southern Potato Salad transformed into macaroni salad with mustard dressing blend with elbow macaroni, pepperoni, peppers, onions, carrots, celery, pickle relish, and southern seasonings. Ingredients: Elbow Macaroni, Salad Dressing, Relish, Sugar, Mustard, Celery, Water, Red Peppers, Onion, Green Peppers, Salt, Spice, Less than 1/10 of 1% Sodium Benzoate, Wheat, Eggs, Soybeans.\nBrand: Amish Wedding, Manufactured by: Harold's Salads, Inc.\nNutrition Facts:\n1 serving per container,\nServing Size: 4oz (113g),\nCalories 270 per serving.\nTotal Fat 14g (18%),\nTotal Carbohydrate 31g (11%),\nCholestrol 15mg (5%),\nSodium 580mg (25%), Protein 4g.
Source: test_file.pdf: p. 1

Lipari Amish Wedding Cranberry Orange Relish (UPC: 049646926107 and Size: 5lb).\nDescription: Fresh cranberries & oranges make this sweet and tangy treat. (Nov-Dec) Ingredients: Cranberries, Oranges, Sugar, Sodium Benzoate, Potassium Sorbate.\nBrand: Amish Wedding, Manufactured by: Harold's Salads, Inc.\nNutrition Facts:\n1 serving per container,\nServing Size: 4oz (113g),\nCalories 160 per serving.\nTotal Fat 0g,\nTotal Carbohydrate 41g (11%),\nCholestrol 0g,\nSodium 15mg (25%), Protein 1g.
Source: test_file.pdf: p. 2

Lipari Rotini Garden Medley Pasta (UPC: 04964926084 and Size: 5lb) are rotini pasta noodles mixed with a medley of carrots, red and green peppers, onions, broccoli florets, red beans, and baby corn in a tasty golden Italian dressing. Some major ingredients include tri-color rotini, italian Dressing, sugar, kidney Beans, corn syrup, onion, red & green Peppers, carrots and baby corn. It has the brand amish wedding and is manufactured by Harold's Salads, Inc. Its packaging is a 5 lb tub with case dimensions 13 5/8 x 6 7/8 x 6 3/16 and has an ambient shelf life of 40 days. For a single serving, its nutritional details are as follows: 160 calories, 4.5g total fat (5% DV), 0.5g saturated fat (3% DV), 0g trans fat, 0mg cholestrol, 250mg sodium (11% DV), 26g carbohydrate (10% DV), 4g dietary fiber (15% DV), 7g total sugars, 4g protein, 0mcg Vitamin D, 20mg Calcium (2% DV), 0.7mg Iron (4% DV) and 110mg Potassium (2% DV).
Source: test_file.pdf: p. 3
=========
FINAL ANSWER: 1. Lipari Amish Wedding Southern Macaroni Salad,\n2. Lipari Amish Wedding Cranberry Orange Relish,\n3. Lipari Rotini Garden Medley Pasta\nSOURCES: test_file.pdf

2)
QUERY: Describe all the items mentioned in the custom lipari file.
=========
Lipari Amish Wedding Southern Macaroni Salad (UPC: 04964926138 and Size: 5lb).\nDescription: Southern Potato Salad transformed into macaroni salad with mustard dressing blend with elbow macaroni, pepperoni, peppers, onions, carrots, celery, pickle relish, and southern seasonings. Ingredients: Elbow Macaroni, Salad Dressing, Relish, Sugar, Mustard, Celery, Water, Red Peppers, Onion, Green Peppers, Salt, Spice, Less than 1/10 of 1% Sodium Benzoate, Wheat, Eggs, Soybeans.\nBrand: Amish Wedding, Manufactured by: Harold's Salads, Inc.\nNutrition Facts:\n1 serving per container,\nServing Size: 4oz (113g),\nCalories 270 per serving.\nTotal Fat 14g (18%),\nTotal Carbohydrate 31g (11%),\nCholestrol 15mg (5%),\nSodium 580mg (25%), Protein 4g.
Source: test_file.pdf: p. 1

Lipari Amish Wedding Cranberry Orange Relish (UPC: 049646926107 and Size: 5lb).\nDescription: Fresh cranberries & oranges make this sweet and tangy treat. (Nov-Dec) Ingredients: Cranberries, Oranges, Sugar, Sodium Benzoate, Potassium Sorbate.\nBrand: Amish Wedding, Manufactured by: Harold's Salads, Inc.\nNutrition Facts:\n1 serving per container,\nServing Size: 4oz (113g),\nCalories 160 per serving.\nTotal Fat 0g,\nTotal Carbohydrate 41g (11%),\nCholestrol 0g,\nSodium 15mg (25%), Protein 1g.
Source: test_file.pdf: p. 2

Lipari Rotini Garden Medley Pasta (UPC: 04964926084 and Size: 5lb) are rotini pasta noodles mixed with a medley of carrots, red and green peppers, onions, broccoli florets, red beans, and baby corn in a tasty golden Italian dressing. Some major ingredients include tri-color rotini, italian Dressing, sugar, kidney Beans, corn syrup, onion, red & green Peppers, carrots and baby corn. It has the brand amish wedding and is manufactured by Harold's Salads, Inc. Its packaging is a 5 lb tub with case dimensions 13 5/8 x 6 7/8 x 6 3/16 and has an ambient shelf life of 40 days. For a single serving, its nutritional details are as follows: 160 calories, 4.5g total fat (5% DV), 0.5g saturated fat (3% DV), 0g trans fat, 0mg cholestrol, 250mg sodium (11% DV), 26g carbohydrate (10% DV), 4g dietary fiber (15% DV), 7g total sugars, 4g protein, 0mcg Vitamin D, 20mg Calcium (2% DV), 0.7mg Iron (4% DV) and 110mg Potassium (2% DV).
Source: test_file.pdf: p. 3
=========
FINAL ANSWER: 1. Lipari Amish Wedding Southern Macaroni Salad (UPC: 04964926138 and Size: 5lb): Southern Potato Salad transformed into macaroni salad with mustard dressing blend with elbow macaroni, pepperoni, peppers, onions, carrots, celery, pickle relish, and southern seasonings.,\n2. Lipari Amish Wedding Cranberry Orange Relish(UPC: 049646926107 and Size: 5lb): Fresh cranberries & oranges make this sweet and tangy treat.,\n3. Lipari Rotini Garden Medley Pasta(UPC: 04964926084 and Size: 5lb): Rotini pasta noodles mixed with a medley of carrots, red and green peppers, onions, broccoli florets, red beans, and baby corn in a tasty golden Italian dressing.\nSOURCES: test_file.pdf
====================================

QUERY: {question}
=========
{doc_summaries}
=========
FINAL ANSWER:
"""

# Final Answer: {
#     'Subject': 'A description of the product named Lipari Amish Wedding Southern Macaroni Salad',
#     'Keywords and headings': 'Lipari Amish Wedding Southern Macaroni Salad, Product Specifications, Brand Amish Wedding, Nutrition Facts',
#     'Summary': 'Lipari Amish Wedding Southern Macaroni Salad (UPC: 04964926138 and Size: 5lb).\nDescription: Southern Potato Salad transformed into macaroni salad with mustard dressing blend with elbow macaroni, pepperoni, peppers, onions, carrots, celery, pickle relish, and southern seasonings. Ingredients: Elbow Macaroni, Salad Dressing, Relish, Sugar, Mustard, Celery, Water, Red Peppers, Onion, Green Peppers, Salt, Spice, Less than 1/10 of 1% Sodium Benzoate, Wheat, Eggs, Soybeans.\nBrand: Amish Wedding, Manufactured by: Harold's Salads, Inc.\nNutrition Facts:\n1 serving per container,\nServing Size: 4oz (113g),\nCalories 270 per serving.\nTotal Fat 14g (18%),\nTotal Carbohydrate 31g (11%),\nCholestrol 15mg (5%),\nSodium 580mg (25%), Protein 4g.'
# }

QA_PROMPT_TEMPLATE_MR2 = """Given the following query and an extracted part of a large document, create a final answer that best answers the query. If you don't know the answer, just say that you don't know. Don't try to make up an answer.

QUERY: List all items mentioned in the lipari file.
============================================
Lipari
Amish Wedding Cranberry Orange Relish
Size
5 lb
UPC
049646926107
Description: Fresh cranberries & oranges make this sweet and tangy treat. (Nov-Dec)
Ingredients: Cranberries, Oranges, Sugar, Sodium Benzoate, Potassium Sorbate.
Product Specifications
Brand Name Amish Wedding
Serving Size:
4 oz
Manufacture Name Herold’s Salads, Inc Shelf Life from date of manufacture 49 days
Kosher Certification:
Kosher Status:
NO
NO
Storage
Shelf Life (Ambient):
Refrigerated
49 days
Packaging
5 lb tub
Kosher Status:
NO
NO
Storage
Shelf Life (Ambient):
Refrigerated
49 days
Packaging
5 lb tub
Net Weight
5 lbs
Case Dimension 13 5/8 x 6 7/8 x 6 3/16
TI/HI
17/6
Country of Origin
USA
Nutrition
Gross Weight 10.85 lbs
XI
Saladsin
inc.
Nutrition
Facts
1 serving per container
Serving size
4 oz (113g)
Calories 160
per serving
Amount per serving
Total Fat Og
Saturated Fat Og
Trans Fat Og
Cholesterol Omg
Sodium 15mg
Vitamin D Omcg 0%
●
% Daily Value *
0%
0%
0%
1%
Amount per serving
Total Carbohydrate 41g
Vitamin D Omcg 0%
●
% Daily Value *
0%
0%
0%
1%
Amount per serving
Total Carbohydrate 41g
Dietary Fiber 3g
Total Sugars 36g
Includes 31g Added Sugars
Protein 1g
Calcium 20mg 2% • Iron 0.2mg 0%
% Daily Value *
15%
9%
.
63%
Potassium 90mg 2%
============================================
FINAL ANSWER: Lipari Amish Wedding Cranberry Orange Relish


QUERY: List and describe all items/products mentioned in the lipari file
============================================
Size
UPC
5 lb 04964926077
Lipari
Mixed Bean Salad
Description: Green, waxed, red kidney, and garbanzo beans, with celery slices, red peppers, carrots, in a sweet
and sour dressing.
Ingredients: Green Beans (Cut Green Beans, Water, Salt, and Zinc Chloride for Stabilization of Color), Wax Beans
(Wax Beans), Kidney Beans (Kidney Beans, High Fructose Corn Syrup, Corn Syrup, Calcium Chloride, EDTA),
Sugar, Celery, Vinegar, Onions, Garbanzo Beans (Garbanzo Beans, Disodium EDTA), Red Peppers (Peppers, and
Citric Acid), Carrots, Corn Oil.
Product Specifications
Brand Name
Amish Wedding
Serving Size:
4 oz
Manufacture Name Herold’s Salads, Inc
Shelf Life from date of
manufacture
49 days
Kosher Certification:
Kosher Status:
Packaging
NO
NO
5 lb tub
Storage
Refrigerated
Shelf Life (Ambient):
0 days
Net Weight
Case Dimension
13 5/8 x 6 7/8 x 6 3/16
Gross Weight
5 lbs
10.85 lbs
TI/HI
0 days
Net Weight
Case Dimension
13 5/8 x 6 7/8 x 6 3/16
Gross Weight
5 lbs
10.85 lbs
TI/HI
Country of Origin
17/6
USA
Nutrition
XI
Herolds
Saladsin
Nutrition
Facts
1 serving per container
Serving size
4 oz (113g)
Calories 160
per serving
Amount per serving
Total Fat 1.5g
Saturated Fat Og
Trans Fat Og
Cholesterol 0mg
Sodium 340mg
Vitamin D Omcg 0%
●
% Daily Value *
2%
0%
Amount per serving
Total Carbohydrate 33g
Dietary Fiber 6g
Total Sugars 21g
% Daily Value
12%
21%
0%
Includes Og Added Sugars
Dietary Fiber 6g
Total Sugars 21g
% Daily Value
12%
21%
0%
Includes Og Added Sugars
Protein 4g
15%
Calcium 80mg 6% • Iron 1.7mg 10% • Potassium 500mg 10%
0%
============================================
FINAL ANSWER: Lipari Mixed Bean Salad with UPC 04964926077 and size 5 lb: Green, waxed, red kidney, and garbanzo beans, with celery slices, red peppers, carrots, in a sweet and sour dressing.


QUERY: Summarise the custom lipari file.
============================================
Lipari
Rotini Garden Medley Pasta Salad
5 lb
04964926084
Description: Rotini pasta noodles mixed with a medley of carrots, red and green peppers, onions, broccoli florets,
red beans, and baby corn in a tasty golden Italian dressing.
Ingredients: Tri-Color Rotini (Water, Pasta (Semolina, Dried Spinach, Dried Tomato, Niacin, Iron (Ferrous Sulfate),
Thiamine Mononitrate, Riboflavin, Folic Acid)), Italian Dressing (Soybean Oil, Water, Distilled Vinegar, High Fructose
Corn Syrup, Salt, Contains 2% or less of: Dried Onion, Dried Garlic, Xanthan Gum, Dried Red Bell Pepper, Spices,
Yellow 5, Yellow 6, Propylene Glycol Alginate, Lemon Juice Concentrate, Calcium Disodium EDTA (To Protect
Flavor), Paprika (Color)), Sugar, Kidney Beans (Red Kidney Beans, Water, Salt, Calcium Chloride, Disodium EDTA
(To Preserve Color)), Ketchup (Tomato Concentrate (Water, Tomato Paste), Corn Syrup, Vinegar, Salt, Onion, Natural
Flavorings, Garlic), Onion, Green Peppers, Carrots, Baby Corn (Baby Corn, Water, Salt, Citric Acid), Broccoli, Red
Pepper (Peppers, Water, Sugar, Salt, Citric Acid), Water, Less than 1/10 of 1% Sodium Benzoate (Preservative).
CONTAINS: WHEAT
Product Specifications
Brand Name Amish Wedding
Manufacture Name Herold’s Salads, Inc Shelf Life from date of manufacture
Kosher Certification:
Kosher Status:
Packaging
NO
NO
5 lb tub
Serving Size:
Storage
4 oz
days
Refrigerated
Shelf Life (Ambient):
Packaging
NO
NO
5 lb tub
Serving Size:
Storage
4 oz
days
Refrigerated
Shelf Life (Ambient):
40 days
Net Weight
5 lbs
Case Dimension 13 5/8 x 6 7/8 x 6
3/16
TI/HI
17/6
Country of Origin
USA
Nutrition
Gross Weight 10.85 lbs
XI
Herolds
Saladsin
inc.
Nutrition
Facts
1 serving per container
Serving size
4 oz (113g)
Calories 160
per serving
Amount per serving
Total Fat 4.5g
Saturated Fat 0.5g
Trans Fat Og
Cholesterol Omg
Sodium 250mg
Vitamin D Omcg 0%
●
% Daily Value *
5%
3%
0%
11%
Calcium 20mg 2%
Cholesterol Omg
Sodium 250mg
Vitamin D Omcg 0%
●
% Daily Value *
5%
3%
0%
11%
Calcium 20mg 2%
Amount per serving
Total Carbohydrate 26g
Dietary Fiber 4g
Total Sugars 7g
Includes Og Added Sugars
Protein 4g
●
% Daily Value *
10%
15%
0%
Iron 0.7mg 4% • Potassium 110mg 2%
============================================
FINAL ANSWER: Lipari Rotini Garden Medley Pasta (UPC: 04964926084 and Size: 5lb) are rotini pasta noodles mixed with a medley of carrots, red and green peppers, onions, broccoli florets, red beans, and baby corn in a tasty golden Italian dressing. Some major ingredients include tri-color rotini, italian Dressing, sugar, kidney Beans, corn syrup, onion, red & green Peppers, carrots and baby corn. It has the brand amish wedding and is manufactured by Harold's Salads, Inc. Its packaging is a 5 lb tub with case dimensions 13 5/8 x 6 7/8 x 6 3/16 and has an ambient shelf life of 40 days. For a single serving, its nutritional details are as follows: 160 calories, 4.5g total fat (5% DV), 0.5g saturated fat (3% DV), 0g trans fat, 0mg cholestrol, 250mg sodium (11% DV), 26g carbohydrate (10% DV), 4g dietary fiber (15% DV), 7g total sugars, 4g protein, 0mcg Vitamin D, 20mg Calcium (2% DV), 0.7mg Iron (4% DV) and 110mg Potassium (2% DV). 

QUERY: {question}
============================================
{context}
============================================
FINAL ANSWER:"""

QUES_CLASS_PROMPT = """Classify the given query into ONLY ONE of 4 classes [summarise, list, describe, keyword extraction, topic extraction].

Query: List and describe all the items mentioned in the lipari file.
Class: describe

Query: List all items mentioned in the smithfield manual.
Class: list

Query: What is the UCL records file about?
Class: topic extraction

Query: Summarise the content of the case demand file.
Class: summarise

Query: Can you summarise the main points of the case ready manual?
Class: summarise

Query: What are the key themes or topics discussed in this file?
Class: topic extraction

Query: Extract and list the major keywords and entities from this file.
Class: keyword extraction

Query: Can you generate a list of keywords related to this file?
Class: keyword extraction

Query: {query}
Class:"""

CLASS_TO_INST = {
    "summarise": """1. Summarise the main points of this text.""",
    "list": """1. From the query, identify the type of entities you need to extract e.g. 'items' likely means consumer food products  e.g. southern potato salad, bone-in centre cut.
2. Extract and return the named entities of the type identified in step 1.""",
    "describe": """1. From the query, identify the type of entities you need to extract e.g. 'items' likely means consumer food products e.g. southern potato salad, bone-in centre cut
2. Extract the named entities of the type identified in step 1.
3. Give a short description from the given context of each entity extracted in step 2.""",
    "keyword extraction": "1. Extract all frequently occuring keywords present in the text.",
    "topic extraction": "1. Identify the key themes or topics of this text."
}

QA_PROMPT_TEMPLATE_MR = """Given the following portion of a large document and the query, use the given instructions to generate an output.
Think step by step and remember not to try to make anything up. 

Query: {question}
Instructions
{instructions}
================================
{context}
================================
Output:
"""

QA_PROMPT_TEMPLATE = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
If you don't know the answer, just say that you don't know and ask the user to provide more information. Don't try to make up an answer.
Your answer should be detailed and elaborate and leave no room for confusion or doubt.
Your answer should be both easy and enjoyable to read with good use of whitespace.
ALWAYS return a "SOURCES" part in your answer.

QUESTION: Which state/country's law governs the interpretation of the contract?
=========
Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.
Source: 28-pl
Content: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.\n\n11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation  in force of the remainder of the term (if any) and this Agreement.\n\n11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any  kind between the parties.\n\n11.9 No Third-Party Beneficiaries.
Source: 30-pl
Content: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,
Source: 4-pl
=========
FINAL ANSWER: This Agreement is governed by English law. Thank you for asking! If you want to ask something else, I'm always happy to assist! 
SOURCES: 28-pl

QUESTION: What did the president say about Michael Jackson?
=========
Content: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\nTonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. \n\nWith a duty to one another to the American people to the Constitution. \n\nAnd with an unwavering resolve that freedom will always triumph over tyranny. \n\nSix days ago, Russia’s Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways. But he badly miscalculated. \n\nHe thought he could roll into Ukraine and the world would roll over. Instead he met a wall of strength he never imagined. \n\nHe met the Ukrainian people. \n\nFrom President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world. \n\nGroups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland.
Source: 0-pl
Content: And we won’t stop. \n\nWe have lost so much to COVID-19. Time with one another. And worst of all, so much loss of life. \n\nLet’s use this moment to reset. Let’s stop looking at COVID-19 as a partisan dividing line and see it for what it is: A God-awful disease.  \n\nLet’s stop seeing each other as enemies, and start seeing each other for who we really are: Fellow Americans.  \n\nWe can’t change how divided we’ve been. But we can change how we move forward—on COVID-19 and other issues we must face together. \n\nI recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner, Officer Jason Rivera. \n\nThey were responding to a 9-1-1 call when a man shot and killed them with a stolen gun. \n\nOfficer Mora was 27 years old. \n\nOfficer Rivera was 22. \n\nBoth Dominican Americans who’d grown up on the same streets they later chose to patrol as police officers. \n\nI spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their mission to restore the trust and safety every community deserves.
Source: 24-pl
Content: And a proud Ukrainian people, who have known 30 years  of independence, have repeatedly shown that they will not tolerate anyone who tries to take their country backwards.  \n\nTo all Americans, I will be honest with you, as I’ve always promised. A Russian dictator, invading a foreign country, has costs around the world. \n\nAnd I’m taking robust action to make sure the pain of our sanctions  is targeted at Russia’s economy. And I will use every tool at our disposal to protect American businesses and consumers. \n\nTonight, I can announce that the United States has worked with 30 other countries to release 60 Million barrels of oil from reserves around the world.  \n\nAmerica will lead that effort, releasing 30 Million barrels from our own Strategic Petroleum Reserve. And we stand ready to do more if necessary, unified with our allies.  \n\nThese steps will help blunt gas prices here at home. And I know the news about what’s happening can seem alarming. \n\nBut I want you to know that we are going to be okay.
Source: 5-pl
Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more. \n\nA unity agenda for the nation. \n\nWe can do this. \n\nMy fellow Americans—tonight , we have gathered in a sacred space—the citadel of our democracy. \n\nIn this Capitol, generation after generation, Americans have debated great questions amid great strife, and have done great things. \n\nWe have fought for freedom, expanded liberty, defeated totalitarianism and terror. \n\nAnd built the strongest, freest, and most prosperous nation the world has ever known. \n\nNow is the hour. \n\nOur moment of responsibility. \n\nOur test of resolve and conscience, of history itself. \n\nIt is in this moment that our character is formed. Our purpose is found. Our future is forged. \n\nWell I know this nation.
Source: 34-pl
=========
FINAL ANSWER: In the content provided, there was no mention of the president mentioning Michael Jackson. If you meant something else, please try refining your query or try giving me more information. I'll be happy to assist. Thank you!
SOURCES:

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""

QA_PROMPT_TEMPLATE_FINAL = """Given the following extracted parts of a long document, merge all these documents into a single concise answers. If the documents contain a `SOURCES` part, include that in your final answer.

=========
{summaries}
=========
FINAL ANSWER:"""

DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {question}\n"
)

DEFAULT_REFINE_PROMPT_TMPL = """The original question is as follows: {question}
We have provided an existing answer, including sources: {existing_answer}
We have the opportunity to refine the existing answer (only if needed) with some more context below.
------------
{context}
------------
Given the new context, refine the original answer to better answer the question. 
If you do update it, please update the sources as well. If the context isn't useful, return the original answer."""
