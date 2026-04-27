from api_bank.apis.api import API
from deep_translator import GoogleTranslator

class Translate(API):

    def __init__(self, init_database=None) -> None:
        if init_database != None:
            self.database = init_database
        else:
            self.database = {}

    def translate(self, text: str, **kwargs) -> str:
        src_lang = kwargs.get('src_lang', 'auto').lower()
        tgt_lang = kwargs.get('tgt_lang', 'en').lower()

        if "chinese" in src_lang:
            src_lang = "zh-TW" if "traditional" in src_lang else "zh-CN"
        if "chinese" in tgt_lang:
            tgt_lang = "zh-TW" if "traditional" in tgt_lang else "zh-CN"
            
        try:
            translated_text = GoogleTranslator(source=src_lang, target=tgt_lang).translate(text)
            return translated_text
        except Exception as e:
            raise Exception(f'Translation failed: {str(e)}')