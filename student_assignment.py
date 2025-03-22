import datetime
import chromadb
import traceback

import pandas as pd
import time

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"
csv_file = "COA_OpenData.csv"


def generate_hw01():
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=gpt_emb_config['api_key'],
        api_base=gpt_emb_config['api_base'],
        api_type=gpt_emb_config['openai_type'],
        api_version=gpt_emb_config['api_version'],
        deployment_id=gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )

    # df = pd.read_csv(csv_file)

    # for idx, row in df.iterrows():
    #     metadata = {
    #         "file_name": csv_file,
    #         "name": row.get("Name", ""),
    #         "type": row.get("Type", ""),
    #         "address": row.get("Address", ""),
    #         "tel": row.get("Tel", ""),
    #         "city": row.get("City", ""),
    #         "town": row.get("Town", ""),
    #         "date": int(time.mktime(datetime.datetime.strptime(row.get("CreateDate", "1970-01-01"), "%Y-%m-%d").timetuple()))
    #     }

    #     document = row.get("HostWords", "")

    #     collection.add(
    #         ids=[str(idx)],
    #         documents=[document],
    #         metadatas=[metadata]
    #     )

    return collection


def generate_hw02(question, city, store_type, start_date, end_date):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=gpt_emb_config['api_key'],
        api_base=gpt_emb_config['api_base'],
        api_type=gpt_emb_config['openai_type'],
        api_version=gpt_emb_config['api_version'],
        deployment_id=gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    where_conditions = []
    if city:
        if isinstance(city, list) and len(city) > 1:
            where_conditions.append({"$or": [{"city": c} for c in city]})
        else:
            where_conditions.append(
                {"city": city[0] if isinstance(city, list) else city})
    if store_type:
        if isinstance(store_type, list) and len(store_type) > 1:
            where_conditions.append({"$or": [{"type": t} for t in store_type]})
        else:
            where_conditions.append(
                {"type": store_type[0] if isinstance(store_type, list) else store_type})
    if start_date:
        where_conditions.append(
            {"date": {"$gte": int(start_date.timestamp())}})
    if end_date:
        where_conditions.append(
            {"date": {"$lte": int(end_date.timestamp())}})  # 小於等於 end_date

    where_filter = {"$and": where_conditions} if where_conditions else None
    # print(where_filter)

    results = collection.query(
        query_texts=[question],
        n_results=10,
        where= where_filter)
    
    filtered_results = []
    for doc, meta, score in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        if score < 0.2:  
            filtered_results.append((meta.get("name"), score))

    filtered_results.sort(key=lambda x: x[1])  
    return [name for name, _ in filtered_results]


def generate_hw03(question, store_name, new_store_name, city, store_type):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=gpt_emb_config['api_key'],
        api_base=gpt_emb_config['api_base'],
        api_type=gpt_emb_config['openai_type'],
        api_version=gpt_emb_config['api_version'],
        deployment_id=gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    results = collection.query(
        query_texts=[question],
        n_results=10,
        where= {"name" : store_name})
    # print(results)
    if (len(results['ids'][0]) != 0):
        store_metadata = results["metadatas"][0][0]
        store_metadata["new_store_name"] = new_store_name

        collection.update(
            ids=[results["ids"][0][0]],metadatas=[store_metadata])
        
    where_conditions = []
    if city:
        if isinstance(city, list) and len(city) > 1:
            where_conditions.append({"$or": [{"city": c} for c in city]})
        else:
            where_conditions.append(
                {"city": city[0] if isinstance(city, list) else city})
    if store_type:
        if isinstance(store_type, list) and len(store_type) > 1:
            where_conditions.append({"$or": [{"type": t} for t in store_type]})
        else:
            where_conditions.append(
                {"type": store_type[0] if isinstance(store_type, list) else store_type})
    where_filter = {"$and": where_conditions} if where_conditions else None
    results = collection.query(
        query_texts=[question],
        n_results=10,
        where= where_filter)
    
    filtered_results = []
    for doc, meta, score in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        if score < 0.2:  
            store_display_name = meta.get("new_store_name", meta.get("name"))
            filtered_results.append((store_display_name, score))

    filtered_results.sort(key=lambda x: x[1])  
    return [name for name, _ in filtered_results]


def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=gpt_emb_config['api_key'],
        api_base=gpt_emb_config['api_base'],
        api_type=gpt_emb_config['openai_type'],
        api_version=gpt_emb_config['api_version'],
        deployment_id=gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )

    return collection


if __name__ == '__main__':
    # demo("我想要找有關茶餐點的店家")
    # collection = generate_hw01()
    # data = collection.get(
    #     limit=1, include=["embeddings", "documents", "metadatas"])
    # print(data)
    # result = generate_hw02("我想要找有關茶餐點的店家", ["宜蘭縣", "新北市"], ["美食"], datetime.datetime(
    #     2024, 4, 1), datetime.datetime(2024, 5, 1))
    # print(result)
    result = generate_hw03("我想要找南投縣的田媽媽餐廳，招牌是蕎麥麵", "耄饕客棧", "田媽媽（耄饕客棧）", ["南投縣"], ["美食"])
    print(result)