import sqlite3
import logging
import time
import pandas as pd 

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('Database.log')
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)

# stream_handler = logging.StreamHandler()

# logger.addHandler(stream_handler)
logger.addHandler(file_handler)

class Database(object):
    def __init__(self, TABLE_NAME="DATA", table_if_exists="append", db='financial.db'):
        self.TABLE_NAME = TABLE_NAME
        self.table_if_exists = table_if_exists
        self.db = db
    
    def create_tables(self, df):
        try:
            conn = sqlite3.connect(self.db) # connect to sqlite3 database
            cursor = conn.cursor() #create a cursor which is used to call execute to create tables. 
            logger.debug('Connected to database successfully...')
            
            cursor.execute(f'''CREATE TABLE IF NOT EXISTS {self.TABLE_NAME}(
                                  TRADE_DT TEXT NOT NULL, 
                                  TRADE_TIME TEXT NOT NULL, 
                                  UNDLY TEXT, 
                                  UNDLY_PRICE REAL NOT NULL, 
                                  EXP_DT TEXT NOT NULL, 
                                  STRIKE INTEGER NOT NULL, 
                                  TYPE TEXT NOT NULL, 
                                  OPTION_REF TEXT NOT NULL, 
                                  MID REAL NOT NULL, 
                                  DTE INTEGER NOT NULL,
                                  RF_RATE REAL NOT NULL, 
                                  DIV_RATE REAL NOT NULL,
                                  POS_SIZE INTEGER NOT NULL, 
                                  Multiplier INTEGER NOT NULL,
                                  IV REAL,
                                  FAIRVALUE REAL, 
                                  DELTA REAL, 
                                  GAMMA REAL, 
                                  THETA REAL,
                                  VEGA REAL, 
                                  WTVEGA REAL,
                                  VOMMA REAL, 
                                  VANNA REAL,
                                  CHARM REAL,
                                  ZOMMA REAL, 
                                  COLOR REAL, 
                                  SPEED REAL, 
                                  DUAL_DELTA REAL, 
                                  DECAY_RATE REAL

                );''')

            df = df[['TRADE_DT','TRADE_TIME','UNDLY','UNDLY_PRICE','EXP_DT','STRIKE','TYPE','OPTION_REF',
                     'MID', 'DTE', 'RF_RATE', 'DIV_RATE', 'POS_SIZE', 'Multiplier', 'IV','FAIRVALUE','DELTA',
                    'GAMMA','THETA','VEGA','WTVEGA','VOMMA', 'VANNA', 'CHARM','ZOMMA', 'COLOR','SPEED',
                    'DUAL_DELTA','DECAY_RATE']]
            #Write dataframe to database. 
            try:
                df.to_sql(self.TABLE_NAME, conn, if_exists=self.table_if_exists, index=False)
            except Exception as e: 
                logger.error(e)
            else:
                conn.commit()
                logger.debug('Written to database successfully...')
                logger.debug('Database closed...')
                conn.close()
                
        except Exception as e:
            logger.exception(e)
                    
    def load_data(self):
        t1 = time.perf_counter()
        sql_data = f'SELECT * FROM {self.TABLE_NAME}'

        conn = sqlite3.connect(self.db)
        logger.debug('Connected to database successfully...')
        
        for DATA in pd.read_sql(sql_data, conn, chunksize=100000):
            return DATA
        
        
        
        
        
        
        