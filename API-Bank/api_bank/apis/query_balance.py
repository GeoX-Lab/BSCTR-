from api_bank.apis.api import API
from api_bank.apis.check_token import CheckToken 

class QueryBalance(API):
    description = 'This API queries the balance of a given user.'

    input_parameters = {
        "token": {'type': 'str', 'description': 'The token of the user.'},
    }
    output_parameters = {
        "balance": {'type': 'float', 'description': 'The balance of the user.'},
    }
    database_name = 'Bank'


    def __init__(self, init_database=None, token_checker=None) -> None:
        if init_database != None:
            self.database = init_database
        else:
            self.database = {}
        
        # 【修复2】移除 assert 强制校验，当外部没有传入校验器时，自动实例化一个默认的
        if token_checker is None:
            self.token_checker = CheckToken()
        else:
            self.token_checker = token_checker
    
    def call(self, token: str) -> dict:
        input_parameters = {
            'token': token,
        }
        try:
            balance = self.query_balance(token)
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
                'output': balance,
                'exception': None,
            }

    def query_balance(self, token: str) -> float:
        #check if the token is correct
        try:
            # 【修复3顺手优化】原代码 token.strip() 是无效操作，因为字符串是不可变的，必须重新赋值
            token = token.strip()
            username = self.token_checker.check_token(token)
        except Exception as e:
            raise Exception('The token is incorrect.')
        
        #check if the username has an account
        if username not in self.database:
            raise Exception('The user does not have an account.')
        
        return self.database[username]['balance']
    
    def check_api_call_correctness(self, response, groundtruth) -> bool:
        response_token = response['input']['token'].strip()
        groundtruth_token = groundtruth['input']['token'].strip()

        response_balance = response['output']
        groundtruth_balance = groundtruth['output']

        if response_token == groundtruth_token and response_balance == groundtruth_balance and response['exception'] == groundtruth['exception']:
            return True
        else:
            return False