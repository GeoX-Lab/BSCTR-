from api_bank.apis.api import API
from rank_bm25 import BM25Okapi
import numpy as np
import nltk
try:
    from nltk.tokenize import word_tokenize
except:
    nltk.download('punkt')
    from nltk.tokenize import word_tokenize

class SearchEngine(API):
    description = 'This API searches for a given keyword for search engine.'
    input_parameters = {
        "keyword": {'type': 'str', 'description': 'The keyword to search.'},
    }
    output_parameters = {
        "results": {'type': 'list', 'description': 'The list of results.'},
    }
    database_name = 'SearchEngine'

    def __init__(self, init_database=None) -> None:
        if init_database != None:
            self.database = init_database
        else:
            self.database = {}
            
        # 【修复 1】：增加判断，兼容框架的无参自动扫描注册逻辑
        if 'tokenized_documents' in self.database:
            self.bm25 = BM25Okapi(self.database['tokenized_documents'])
        else:
            self.bm25 = None # 注册阶段只需元信息，不需要真正初始化搜索引擎

    def call(self, keyword: str) -> dict:
        input_parameters = {
            'keyword': keyword,
        }
        try:
            results = self.search(keyword)
        except Exception as e:
            exception = str(e)
            return {
                'api_name': self.__class__.__name__,
                'input': input_parameters,
                'output': None,
                'exception': exception,
            }
        else:
            return {
                'api_name': self.__class__.__name__,
                'input': input_parameters,
                'output': results,
                'exception': None,
            }

    def search(self, keyword: str) -> list:
        keyword = keyword.lower().strip()
        query = word_tokenize(keyword) # keyword.split()
        
        # 确保真正的评测中 bm25 被正确初始化了
        if self.bm25 is None:
            raise Exception("Search engine is not properly initialized with documents.")
            
        rankings = np.argsort(-np.array(self.bm25.get_scores(query)))
        if len(rankings) > 2:
            rankings = rankings[:2]
        results = [self.database["raw_documents"][i] for i in rankings]
        return results
    
    def check_api_call_correctness(self, response, groundtruth) -> bool:
        if response['api_name'] != groundtruth['api_name']:
            return False
        if response['exception'] != groundtruth['exception']:
            return False
            
        # 【修复 2】：原代码如果有 None 的情况，会导致 response_output 变量未定义就直接被后面的 if 比较
        response_output = None
        groundtruth_output = None
        
        if response['output'] != None:
            response_output = sorted(response['output'], key=lambda x: x['title']+x['abstract'], reverse=True)
        if groundtruth['output'] != None:
            groundtruth_output = sorted(groundtruth['output'], key=lambda x: x['title']+x['abstract'], reverse=True)
            
        if response_output != groundtruth_output:
            return False
        return True