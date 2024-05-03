import os
import re
from datetime import date
from functools import partial
from io import StringIO
from itertools import islice
from typing import Optional, Union

import neo4j
import numpy as np
import pandas as pd


def rename_column(df,prev_name,new_name):
    df[new_name] = df[prev_name]
    del df[prev_name]
# num2int = lambda a: int(a) if (a != '' and not a is None) else -100
def num2int(a,default=0):
    if a != '' and not a is None:
        try:
            return np.uint32(a)
        except:
            return a.strip(' *')
    else:
        return default

strc = lambda a: ' '.join([ai.capitalize() for ai in a.split(' ')]).strip(' *')

def num2str(num):
    try:
        return str(num)
    except Exception as e:
        # print(e)
        return num
    

# num2float = lambda a: np.float32(f'{a}'.replace(',','') if (a != '' and not a is None) else 0)
def num2float(a, default=0):
    if a != '' and not a is None:
        try:
            return np.float32(a.replace(',','').replace('$',''))
        except Exception as e:
            # print(e)
            return a
    else:
        return default


def curr_to_float(num, default=-1.0):
    if isinstance(num, float) or isinstance(num, int):
        return float(num)
    try:
        return float(num.strip('$-'))
    except ValueError:
        if default is None:
            # print(num)
            return num.strip('$-')
        return default
    except AttributeError:
        return float(num)

def percent_to_float(num, default=-1.0):
    if isinstance(num, float) or isinstance(num, int):
        return float(num)
    try:
        return float(num.strip('%-'))
    except ValueError:
        if default is None:
            # print(num)
            return num
        return default
    except AttributeError:
        return float(num)

#+'++'+toString(row.PrevYrNetWgtShp)+'/'+toString(row.PrevYrNetCasesShpC)
CREATE_RECORDS_CYPHER = """WITH {jan:1,feb:2,mar:3,apr:4,may:5,jun:6,jul:7,aug:8,sep:9,oct:10,nov:11,dec:12} AS months
UNWIND $rows as row
MERGE (v:Vendor {number:toInteger(row.VendorNo)})
FOREACH(x in CASE WHEN v.name IS NULL OR NOT (row.VendorName IN v.name) THEN [1] END |
    SET v.name = COALESCE(v.name,[]) + row.VendorName )
MERGE (c:Customer {number:row.CustomerNo}) 
SET
    c.address = row.CustomerAddress,
    c.city = row.CustomerCity,
    c.state = row.CustomerState,
    c.zipcode = row.CustomerZipCode
FOREACH(x in CASE WHEN c.name IS NULL OR NOT (row.CustomerName IN c.name) THEN [1]  END |
    SET c.name = COALESCE(c.name,[]) + row.CustomerName )

MERGE (i:Item {number: toInteger(row.ItemNo)})
SET
    i.description = row.ItemDescription,
    i.brand_name = row.ItemBrand,
    i.name = toString(row.ItemBrand + ' ' + row.ItemDescription),
    i.size = row.ItemSize,
    i.price = COALESCE(i.price, -1),
    i.po_cost = COALESCE(i.po_cost, -1),
    i.po_cost_freight = COALESCE(i.po_cost_freight, -1),
    i.freight = COALESCE(i.freight, 0),
    i.store_cost_cs = COALESCE(i.store_cost_cs, -1),
    i.store_cost_unit = COALESCE(i.store_cost_unit, -1),
    i.est_reg_cost = COALESCE(i.est_reg_cost, -1),
    i.case_weight = COALESCE(i.case_weight, -1),
    i.z8_margin = COALESCE(i.z8_margin, 0)

MERGE (sp:SalesRep {number: toInteger(row.SalesRepNo)})
ON CREATE SET
    sp.name = row.SalesRepName
MERGE (sg:SalesRegion {number: toInteger(row.SalesRegionNo)})
ON CREATE SET
    sg.description = row.SalesRegionDesc
WITH 'R'+'V'+toString(v.number)+'-C'+toString(c.number)+'-I'+toString(i.number)+'-SR'+toString(sg.number)+'-SP'+toString(sp.number)+'D'+toString(months[toLower(substring(trim(row.Month),0,3))])+toString(toInteger(row.Year))+'='+toString(row.CurrentYrNetWgtShp)+'/'+toString(row.CurrentYrNetCasesShp) AS rid
CREATE (r:Record {id: rid})
SET
    r.record_date = date({year: toInteger(row.Year), month: months[toLower(substring(trim(row.Month),0,3))]}),
    r.record_cases = toInteger(row.CurrentYrNetCasesShp),
    r.record_weight = row.CurrentYrNetWgtShp
    
MERGE (c)-[:BOUGHT_ITEM]-(i)
MERGE (v)-[:SOLD_ITEM]-(i)
MERGE (v)-[:SELLER_CUSTOMER]-(c)
MERGE (r)-[:RECORDED_VENDOR]-(v)
MERGE (r)-[:RECORDED_CUSTOMER]-(c)
MERGE (r)-[:RECORDED_ITEM]-(i)
MERGE (r)-[:RECORDED_SALES_REP]-(sp)
MERGE (r)-[:RECORDED_SALES_REGION]-(sg);"""

CREATE_ITEMS_CYPHER = """UNWIND $rows AS row
MERGE (i:Item {number: row.ItemNumber})
SET
    i.description = row.ItemDescription,
    i.brand_name = row.Brand,
    i.name = toString(row.Brand + ' ' + row.ItemDescription),
    i.ven_item_number = row.VendorItem,
    i.size = row.ItemSize,
    i.item_pack = row.CasePack,
    i.category = row.Category,
    i.tenant = row.Tenant,
    i.upc = row.Upc,
    i.price = row.Srp,
    i.po_cost = row.PoCost,
    i.po_cost_freight = row.PoCostAfterFreight,
    i.freight = row.CurrentFreight,
    i.store_cost_cs = row.EstStoreCostCs,
    i.store_cost_unit = row.EstStoreCostPerUnitOrLb,
    i.est_reg_cost = row['EstimatedRegCostZ8Unit'],
    i.case_weight = row.CaseWeightLbs,
    i.z8_margin = row.Zone8Margin,
    i.accrual_at_lipari = row.AccrualAtLipari,
    i.promotional_allowance = row.PromotionalAllowanceApproved,
    i.onediamond_rep = row.OneDiamondRep,
    i.c_store = row.CStoreItems,
    i.category_manager = row.CategoryManager

MERGE (v:Vendor {number: row.VendorNumber})
ON CREATE SET
    v.name = row.VendorName

MERGE (v)-[:SOLD_ITEM]-(i);
"""

CREATE_CUSTOMERS_CYPHER = """UNWIND $rows AS row
MERGE (c:Customer {number: row.CustomerNo})
SET
    c.name = row.CustomerName,
    c.address = row.CustomerAddress
    c.chain = row.ChainName;"""

class CSV2KG:

    def __init__(
        self, 
        url: str, 
        database: str = 'neo4j', 
        username: str = 'neo4j', 
        password: Optional[str] = None,
        batch_size: int = 5000
    ):
        self.batch_size = batch_size

        url = os.environ.get("NEO4J_URI", url) #get_from_env("url", "NEO4J_URI", url)
        username = os.environ.get("NEO4J_USERNAME", username) #get_from_env("username", "NEO4J_USERNAME", username)
        password = os.environ.get("NEO4J_PASSWORD", password) #get_from_env("password", "NEO4J_PASSWORD", password)
        database = os.environ.get("NEO4J_DATABASE", database) #get_from_env("database", "NEO4J_DATABASE", database)
        
        if password is None or username is None:
            auth = None
        elif username is not None and password is not None:
            auth = (username, password)
        else:
            # print(username, password)
            auth = None
        self._driver = neo4j.GraphDatabase.driver(url, auth=auth)
        self._database = database
        # Verify connection
        try:
            self._driver.verify_connectivity()
        except neo4j.exceptions.ServiceUnavailable:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the url is correct"
            )
        except neo4j.exceptions.AuthError:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the username and password are correct"
            )
    def __del__(self):
        self._driver.close()

    @classmethod
    def create_items_vendors_tx(cls, tx:neo4j.Transaction, rows: str):
        tx.run(CREATE_ITEMS_CYPHER, rows=rows)

    @classmethod
    def create_records_tx(cls, tx: neo4j.Transaction, rows: str):
        tx.run(CREATE_RECORDS_CYPHER.strip('\n'), rows=rows) #i.itype = row.ItemType,r.act_record_code = row.ActiveRecordCode

    @classmethod
    def create_customers_tx(cls, tx: neo4j.Transaction, rows: str):
        tx.run(CREATE_CUSTOMERS_CYPHER, rows=rows)

    def create_nodes(self, rows, datatype: int = 0):
        if datatype == 1: # True or non-zero means read records
            tx_fn = self.create_records_tx
            # creation_cypher = CREATE_RECORDS_CYPHER
        elif datatype == 0: # False or zero means read items
            tx_fn = self.create_items_vendors_tx
            # creation_cypher = CREATE_ITEMS_CYPHER
        elif datatype == 2:
            tx_fn = self.create_customers_tx
            # creation_cypher = CREATE_CUSTOMERS_CYPHER
        if len(rows) <= self.batch_size:
            with self._driver.session(database=self._database) as session:
                session.execute_write(tx_fn, rows)
        else:
            stream = iter(rows)
            i = 0
            session = self._driver.session(database=self._database) #
            while True:
                batch = list(islice(stream, self.batch_size))
                if len(batch) > 0:
                    print(f"Rows {i}-{i+len(batch)}")
                    try:
                        session.execute_write(tx_fn, batch)
                        # session.run(creation_cypher, parameters={"rows": batch})
                    except Exception as e:
                        print(e)
                        #for b in batch:
                            #print(b)
                        #break
                    i += len(batch)
                else:
                    break
            session.close()

    def read_items(self, filepath: Optional[str]=None, data: Optional[str]=None, sep=','):
        if filepath is not None and len(filepath) > 0:
            if filepath.endswith('.csv'):
                df_items = pd.read_csv(filepath, sep=sep)
            elif filepath.endswith('.parq'):
                df_items = pd.read_parquet(filepath, sep=sep)
        elif data is not None and len(data) > 0:
            df_items = pd.read_csv(StringIO(data), sep=sep)
        else:
            raise Exception("Either data (CSV string) or filepath must be provided")
        
        df_items.fillna(
            value={
                'Upc': "-1", 
                'CStoreItems': '', 
                'PoCost': -1, 
                'VendorItemNo': '', 
                'CurrentFreight': -1, 
                'Zone8Margin': 0.0, 
                'AccrualAtLipari': "", 
                'PromotionalAllowanceApproved': ""
            },
            inplace=True
        )
        
        df_items['Upc'] = df_items['Upc'].apply(lambda a: int(a.replace(' ','')))
        df_items['Srp'] = df_items['Srp'].apply(curr_to_float)
        df_items['CurrentFreight'] = df_items['CurrentFreight'].apply(curr_to_float, default=0.0)
        df_items['EstStoreCostPerUnitOrLb'] = df_items['EstStoreCostPerUnitOrLb'].apply(curr_to_float, default=0.0)
        df_items['EstimatedRegCostZ8Unit'] = df_items['EstimatedRegCostZ8Unit'].apply(curr_to_float, default=0.0)
        df_items['PoCost'] = df_items['PoCost'].apply(curr_to_float, default=0.0)
        df_items['AccrualAtLipari'] = df_items['AccrualAtLipari'].apply(lambda a: a.replace('$-',''))
        df_items['EstStoreCostCs'] = df_items['EstStoreCostCs'].apply(curr_to_float, default=0.0)
        df_items['PoCostAfterFreight'] = df_items['PoCostAfterFreight'].apply(curr_to_float, default=0.0)
        df_items['PoCostAfterFreightPerUnit'] = df_items['PoCostAfterFreightPerUnit'].apply(curr_to_float, default=0.0)
        df_items['PromotionalAllowanceApproved'] = df_items['PromotionalAllowanceApproved'].apply(lambda a: a.strip('$-'))
        df_items['Zone8Margin'] = df_items['Zone8Margin'].apply(percent_to_float, default=0.0)
        
        # df_items.drop(columns=["How Is Item Priced? Rnd Or Upc'd", "Category.1"], inplace=True)

        # for col in df_items.columns:
        #     ncol = col.replace(' ','').replace('-','').replace('/','')
        #     rename_column(df_items, col, ncol)

        # print(df_items.isna().sum())
        self.create_nodes(df_items.to_dict('records'), datatype=0)

    def read_records(self, filepath: Optional[str]=None, data: Optional[str]=None, stores_filepath: Optional[str] = None, stores_data: Optional[str] = None, start=0, end=-1, sep=','):
    
        if stores_filepath is not None and len(stores_filepath) > 0:
            if stores_filepath.endswith('.csv'):
                df_stores = pd.read_csv(stores_filepath)
            elif stores_filepath.endswith('.parq'):
                df_stores = pd.read_parquet(stores_filepath)
        elif stores_data is not None and len(stores_data) > 0:
            df_stores = pd.read_csv(StringIO(stores_data))
        else:
            df_stores = None

        if df_stores is not None:
            df_stores.set_index("CustomerNo", inplace=True)
        if filepath is not None and len(filepath) > 0:
            
            converters = {'MONTH':strc,'YEAR':partial(num2int,default=2024),'VendorNo':num2int,'VendorName':strc,
            'SalesRegionNo':num2int,'SalesRegionDesc':strc,'CustomerNo':num2str,'CustomerName':strc,
            'CustomerAddress':strc,'CustomerCity':strc,'CustomerState':lambda a:a.upper(),'CustomerZipCode':strc,
            'SalesRepNo':num2int,'SalesRepName':strc,'ItemNo':num2int,'ItemDescription':strc,'ItemBrand':strc,
            'ItemPack':num2int,'ItemSize':strc,
            'CurrentYrNetCasesShp':num2float,'PrevYrNetCasesShpC':num2float,'DiffNetCasesShpD':num2float,
            'CurrentYrNetWgtShp':num2float,'PrevYrNetWgtShp':num2float,'DiffNetWgtShpD':num2float
            } #,'ItemType':strc,'ActiveRecordCode':strc
            if filepath.endswith('.csv'):
                df_store_recs = pd.read_csv(filepath, sep=sep, converters=converters, usecols=list(converters.keys()))
            elif filepath.endswith('.parq'):
                df_store_recs = pd.read_parquet(filepath)
                df_store_recs['MONTH'] = df_store_recs['MONTH'].apply(strc)
                df_store_recs['YEAR'] = df_store_recs['YEAR'].apply(partial(num2int, default=2023))
                df_store_recs['VendorNo'] = df_store_recs['VendorNo'].apply(num2int)
                df_store_recs['VendorName'] = df_store_recs['VendorName'].apply(strc)
                df_store_recs['CustomerNo'] = df_store_recs['CustomerNo'].apply(num2int)
                df_store_recs['CustomerAddress'] = df_store_recs['CustomerAddress'].apply(strc)
                df_store_recs['CustomerCity'] = df_store_recs['CustomerCity'].apply(strc)
                df_store_recs['CustomerName'] = df_store_recs['CustomerName'].apply(strc)
                df_store_recs['CustomerZipCode'] = df_store_recs['CustomerZipCode'].apply(strc)
                df_store_recs['CustomerState'] = df_store_recs['CustomerState'].apply(lambda a: a.upper())
                df_store_recs['ItemNo'] = df_store_recs['ItemNo'].apply(num2int)
                df_store_recs['ItemBrand'] = df_store_recs['ItemBrand'].apply(strc)
                if df_stores is not None:
                    df_store_recs['ChainName'] = df_store_recs['CustomerNo'].apply(
                        lambda n: df_stores.loc[n, "ChainName"] if n in df_stores.index else "INDEPENDENTS"
                    )
                df_store_recs['ChainName'] = df_store_recs['ChainName'].apply(strc)
                # df_store_recs['ItemType'] = df_store_recs['ItemType'].apply(strc)
                df_store_recs['ItemSize'] = df_store_recs['ItemSize'].apply(strc)
                df_store_recs['ItemPack'] = df_store_recs['ItemPack'].apply(num2int)
                df_store_recs['ItemDescription'] = df_store_recs['ItemDescription'].apply(strc)
                df_store_recs['SalesRegionNo'] = df_store_recs['SalesRegionNo'].apply(num2int)
                df_store_recs['SalesRegionDesc'] = df_store_recs['SalesRegionDesc'].apply(strc)
                df_store_recs['SalesRepNo'] = df_store_recs['SalesRepNo'].apply(num2int)
                df_store_recs['SalesRepName'] = df_store_recs['SalesRepName'].apply(strc)
                df_store_recs['CurrentYrNetCasesShp'] = df_store_recs['CurrentYrNetCasesShp'].apply(num2float)
                df_store_recs['PrevYrNetCasesShpC'] = df_store_recs['PrevYrNetCasesShpC'].apply(num2float)
                df_store_recs['CurrentYrNetWgtShp'] = df_store_recs['CurrentYrNetWgtShp'].apply(num2float)
                df_store_recs['PrevYrNetWgtShp'] = df_store_recs['PrevYrNetWgtShp'].apply(num2float)
        elif data is not None and len(data) > 0:
            converters = {'MONTH':strc,'YEAR':partial(num2int,default=2023),'VendorNo':num2int,'VendorName':strc,
            'SalesRegionNo':num2int,'SalesRegionDesc':strc,'CustomerNo': num2str,'CustomerName':strc,
            'CustomerAddress':strc,'CustomerCity':strc,'CustomerState':lambda a: str(a).upper(),'CustomerZipCode':strc,
            'SalesRepNo':num2int,'SalesRepName':strc,'ItemNo':num2int,'ItemDescription':strc,'ItemBrand':strc,
            'ItemPack':num2int,'ItemSize':strc,
            'CurrentYrNetCasesShp':num2float,'PrevYrNetCasesShpC':num2float,'DiffNetCasesShpD':num2float,
            'CurrentYrNetWgtShp':num2float,'PrevYrNetWgtShp':num2float,'DiffNetWgtShpD':num2float
            } #'ItemType':strc,'ActiveRecordCode':strc,
            df_store_recs = pd.read_csv(StringIO(data), converters=converters, usecols=list(converters.keys()), sep=sep)
            if df_stores is not None:
                df_store_recs['ChainName'] = df_store_recs['CustomerNo'].apply(
                    lambda n: df_stores.loc[n, "ChainName"] if n in df_stores.index else "INDEPENDENTS"
                )
        else:
            raise Exception("At least 1 of `filepath(path to file)` or `data(CSV string)` must be provided.")
        
        # for col in df_store_recs.columns:
        #     if str(df_store_recs[col].dtype)== 'object':
        #                df_store_recs[col] = df_store_recs[col].apply(lambda a: a.strip(' *'))
        
        
        rename_column(df_store_recs,'MONTH','Month')
        rename_column(df_store_recs,'YEAR','Year')
        # print(df_store_recs.dtypes)

        if end <= 0 or end > len(df_store_recs):
            end = len(df_store_recs)
        
        if start < 0 or start > len(df_store_recs):
            start = 0
        # print(start,end,end-start)
        recs_data = df_store_recs.iloc[start:end,:].to_dict('records')
        self.create_nodes(recs_data, 1)
        # create_update_rgraph(graph, recs_data, batch_size=5_000)
        return
    
    def read_stores(
            self, 
            filepath: Optional[str] = None, 
            data: Optional[str] = None, 
            start: int = 0, 
            end: int = -1, 
            sep: str = ','
        ):
        if filepath is not None and len(filepath) > 0:
            converters = {"CustomerNo": num2str, "CustomerName": strc, "ChainName": strc, "CustomerAddress": strc}
            if filepath.endswith('.csv'):
                df_stores = pd.read_csv(filepath, sep=sep, converters=converters, usecols=list(converters.keys()))
            elif filepath.endswith('.parq'):
                df_stores = pd.read_parquet(filepath)
                df_stores = df_stores[list(converters.keys())]
                df_stores['CustomerNo'] = df_stores['CustomerNo'].apply(num2str)
                df_stores['CustomerName'] = df_stores['CustomerName'].apply(strc)
                df_stores['ChainName'] = df_stores['ChainName'].apply(strc)
                df_stores['CustomerAddress'] = df_stores['CustomerAddress'].apply(strc)
            else:
                raise ValueError(f"Invalid File Extension for file {filepath} not supported. Only .csv and .parq files are supported")
        
        elif data is not None and len(data) > 0:
            converters = {"CustomerNo": num2str, "CustomerName": strc, "ChainName": strc, "CustomerAddress": strc}
            df_stores = pd.read_csv(StringIO(data), converters=converters, usecols=list(converters.keys()), sep=sep)
        else:
            raise Exception("At least 1 of `filepath(path to file)` or `data(CSV string)` must be provided.")
        
        if end <= 0 or end > len(df_stores):
            end = len(df_stores)
        
        if start < 0 or start > len(df_stores):
            start = 0
        # print(start,end,end-start)
        stores_data = df_stores.iloc[start:end,:].to_dict('records')
        self.create_nodes(stores_data, 2)
        # create_update_rgraph(graph, recs_data, batch_size=5_000)
        return