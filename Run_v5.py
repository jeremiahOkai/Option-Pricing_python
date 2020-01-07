import time 
import pandas as pd
import logging 
import sys
import numpy as np
import os
from pandarallel import pandarallel as pl
import glob
from datetime import datetime 


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) 

formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
file_handler =logging.FileHandler('run.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# pl.initialize() #unlock this when executing new dataset 



from Database_v5 import Database
from Calculators_v2 import Calculators
from PreprocessData import Preprocess
from Strategies_v9 import Strategies

if __name__ == "__main__":
    pp = Preprocess()
    pp.execute_process()

# class Run(object):
#     def __init__(self, filename='./ProcessFiles/output.csv'):
#         self.filename = filename
               
#     @classmethod    
#     def run_calculators(cls, filename):
#         try:  
#             # pl.initialize()
#             logger.debug('Reading data...')
#             logger.debug('Calculating Greeks...')
#             t0 = time.perf_counter()
            
#             db = Database()
#             counter = 0
#             df=pd.read_csv(filename)
#             print(f'calculating sigma...')
#             df['IV'] = df.parallel_apply(Calculators.newton_vol_call_put_div, axis=1)
#             print(f'calculating FAIRVALUE...')
#             df['FAIRVALUE'] = df.parallel_apply(Calculators.euro_vanilla_dividend, axis=1)
#             print(f'calculating DELTA...')
#             df['DELTA'] = df.parallel_apply(Calculators.delta_div, axis=1)
#             print(f'calculating GAMMA...')
#             df['GAMMA'] = df.parallel_apply(Calculators.gamma_div, axis=1)
#             print(f'calculating THETA...')
#             df['THETA'] = df.parallel_apply(Calculators.theta_div, axis=1)
#             print(f'calculating VEGA...')
#             df['VEGA'] = df.parallel_apply(Calculators.vega_div, axis=1)
#             print(f'calculating WTVEGA...')
#             df['WTVEGA'] = df.parallel_apply(Calculators.wtvega, axis=1)
#             print(f'calculating ZOMMA...')
#             df['ZOMMA'] = df.parallel_apply(Calculators.zomma, axis=1)
#             print(f'calculating SPEED...')
#             df['SPEED'] = df.parallel_apply(Calculators.speed, axis=1)
#             print(f'calculating DUAL_DELTA...')
#             df['DUAL_DELTA'] = df.parallel_apply(Calculators.dual_delta, axis=1)
#             print(f'calculating DECAY_RATE...')
#             df['DECAY_RATE'] =df.parallel_apply(Calculators.decay_rate, axis=1)
#             print(f'calculating VOMMA...')
#             df['VOMMA'] =df.parallel_apply(Calculators.vomma, axis=1)
#             print(f'calculating VANNA...')
#             df['VANNA'] = df.parallel_apply(Calculators.vanna, axis=1)
#             print(f'calculating CHARM...')
#             df['CHARM'] =df.parallel_apply(Calculators.charm, axis=1)
#             print(f'calculating COLOR...')
#             df['COLOR'] =df.parallel_apply(Calculators.color, axis= 1)
#             db.create_tables(df)

#             counter +=1
#             t1 = time.perf_counter()
#             logger.debug(f'Finished executing all tasks {(t1 - t0)/3600} hour(s)...')

#         except Exception as e: 
#             logger.exception(e)
 

#     def execute_process(self): 
#         print(
#         """
#         ########################################################################################################

#             Welcome to the Option Pricing Code...
#             Follow the instructions below to use the code successfully. 

#             1. Enter 1 to run an Option Strategy.
#             2. Enter 2 to execute Preprocess - add new option data, calculate greeks and write to the database. 

#         ########################################################################################################
#         """)
#         try: 
#             value = input('Enter a value, either 1 or 2 to continue... ')
#             if not value.strip():
#                 raise ValueError('Field cannot be empty')
#             try:
#                 value = int(float(value))
#             except TypeError as e: 
#                 logger.debug('Integer or float values only')
#                 sys.exit('Exiting program...')
#         except ValueError as e:
#             logger.debug(e)
#             sys.exit('Exiting program...')

#         if value == 1: 
#             print(
#             """
#             ############################################################################################
#                                     1. Choose 1 for Delta Strategy. 
#                                     2. Choose 2 for Distance Strategy. 
#                                     3. Choose 3 for Percentage Strategy. 
#                                     4. Choose 4 for Absolute Strategy. 
#                                     5. Choose 5 for Mid Strategy 
#                                     6. Choose 6 for Mid-Strike Strategy.
#             ############################################################################################
#             """
#             )
#             try: 
#                 strategy = input('Choose an option above to execute strategy... ')
#                 if not strategy.strip():
#                     raise ValueError('Field cannot be empty')
#                 try: 
#                     strategy = int(float(strategy))
#                 except TypeError as e: 
#                     logger.debug('Integer or float values only')
#                     sys.exit('Exiting program...')
#             except ValueError as e:
#                 logger.debug(e)
#                 sys.exit('Exiting program...')

#             print(
#                 '''
#             #############################################################################################
#                     Enter all values separated by commas with no spaces between them. Eg. 1,2,3
#                     Starting and ending dates must be of format = 2010-10-01 (yyyy-mm-dd)
#                     All values must be provided else the program will raise an error and exit. 
#             #############################################################################################
#                 '''
#                 )


#             if strategy == 1: 
#                 print(
#                 '''
#             $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#                                         Execting the Delta Strategy 
#             $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#                 '''
#                 )

#                 try:
#                     delta=input('Enter delta values: ')
#                     if not delta.strip():
#                         raise ValueError('Field cannot be empty')
#                 except ValueError as e:
#                     logger.debug(e)
#                     sys.exit('Exiting program...')
#                 else: 
#                     delta = [int(i) for i in delta.split(',')] 
#                     if len(delta) == 1: 
#                         delta = delta[0]

#                 MID = None 
#                 MID_STRIKE = None 
#                 strike = None 
#                 distance = None 
#                 percent = None 

#             elif strategy == 2: 
#                 print(
#                 '''
#             $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#                                         Execting the Distance Strategy 
#             $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#                 ''')
#             elif strategy == 3: 
#                 print(
#                 '''
#             $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#                                         Execting the Percentage Strategy 
#             $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#                 ''') 

#                 try:
#                     percent=input('Enter percent values: ')
#                     if not percent.strip():
#                         raise ValueError('Field cannot be empty')
#                 except ValueError as e:
#                     logger.debug(e)
#                     sys.exit('Exiting program...')
#                 else: 
#                     percent = [int(i) for i in percent.split(',')] 
#                     if len(percent) == 1: 
#                         percent = percent[0]
#                 MID = None 
#                 MID_STRIKE = None 
#                 strike = None 
#                 distance = None 
#                 delta = None 


#             elif strategy == 4: 
#                 print(
#                 '''
#             $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#                                         Execting the Absolute Strategy 
#             $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#                 ''') 

#                 try:
#                     strike=input('Enter strike values: ')
#                     if not strike.strip():
#                         raise ValueError('Field cannot be empty')
#                 except ValueError as e:
#                     logger.debug(e)
#                     sys.exit('Exiting program...')
#                 else: 
#                     strike = [int(i) for i in strike.split(',')] 
#                     if len(strike) == 1: 
#                         strike = strike[0]

#                 MID = None 
#                 MID_STRIKE = None 
#                 distance = None 
#                 delta = None 
#                 percent = None 

#             elif strategy == 5: 
#                 print(
#                 '''
#             $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#                                         Execting the MID Strategy 
#             $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#                 ''') 

#                 try:
#                     MID=input('Enter MID values: ')
#                     if not MID.strip():
#                         raise ValueError('Field cannot be empty')
#                 except ValueError as e:
#                     logger.debug(e)
#                     sys.exit('Exiting program...')
#                 else: 
#                     MID = [int(i) for i in MID.split(',')] 
#                     if len(MID) == 1: 
#                         MID = MID[0]
#                 MID_STRIKE = None 
#                 strike = None 
#                 distance = None 
#                 delta = None 
#                 percent = None 

#             elif strategy == 6: 
#                 print(
#                 '''
#             $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#                                         Execting the MID-STRIKE Strategy 
#             $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#                 ''') 

#                 try:
#                     MID_STRIKE=input('Enter MID_STRIKE values: ')
#                     if not MID_STRIKE.strip():
#                         raise ValueError('Field cannot be empty')
#                 except ValueError as e:
#                     logger.debug(e)
#                     sys.exit('Exiting program...')
#                 else: 
#                     MID_STRIKE = [int(i) for i in MID_STRIKE.split(',')]
#                     if len(MID_STRIKE) == 1: 
#                         MID_STRIKE = MID_STRIKE[0]
#                 strike = None 
#                 distance = None 
#                 delta = None 
#                 MID = None 
#                 percent = None 

#             try:
#                 leg = input('Enter number of legs: ')
#                 if not leg.strip():
#                     raise ValueError('Field cannot be empty')
#                 try:
#                     leg=int(float(leg))
#                 except ValueError as e: 
#                     logger.debug('Value must be integer or float')
#                     sys.exit('Exiting program...')
#             except ValueError as e:
#                 print(e)
#                 sys.exit('Exiting program...')

#             try:
#                 dte = input('Enter DTE: ')
#                 if not dte.strip():
#                     raise ValueError('Field cannot be empty')
#                 try:
#                     dte=int(float(dte))
#                 except ValueError as e: 
#                     print('Value must be integer or float')
#                     sys.exit('Exiting program...')
#             except ValueError as e:
#                 logger.debug(e)
#                 sys.exit('Exiting program...')
#             else: 
#                 if len(dte) == 1: 
#                     dte = dte[0]

#             try:
#                 INPUT_DAYS = input('Enter INPUT_DAYS: ')
#                 if not INPUT_DAYS.strip():
#                     raise ValueError('Field cannot be empty')
#                 try:
#                     INPUT_DAYS=int(float(INPUT_DAYS))
#                 except ValueError as e: 
#                     logger.debug('Value must be integer or float')
#                     sys.exit('Exiting program...')
#             except ValueError as e:
#                 logger.debug(e)
#                 sys.exit('Exiting program...')

#             try:
#                 sdate= input('Enter start date: ')
#                 edate= input('Enter end date: ')
#                 PC= input('Enter PC: ')
#                 if not sdate.strip() or not edate.strip() or not PC.strip():
#                     raise ValueError('Field cannot be empty')
#                 try:
#                     sd=datetime.strptime(sdate, '%Y-%m-%d')
#                     ed=datetime.strptime(edate, '%Y-%m-%d')
#                 except ValueError as e: 
#                     logger.debug("Incorrect data format, should be YYYY-MM-DD")
#                     sys.exit('Exiting program...')
#             except ValueError as e:
#                 logger.debug(e)
#                 sys.exit('Exiting program...')
#             else: 
#                 PC=PC.upper()

#             try:
#                 user_range=input('Enter delta range to check before opening position: ')
#                 contract=input('Enter contracts: ')
#                 if not user_range.strip() or not contract.strip():
#                     raise ValueError('Field cannot be empty')
#             except ValueError as e:
#                 logger.debug(e)
#                 sys.exit('Exiting program...')
#             else: 
#                 user_range = [float(i) for i in user_range.split(',')] 
#                 contract = [int(i) for i in contract.split(',')] 
#                 if len(contract) == 1: 
#                     contract = contract[0]

#             try:
#                 P_DELTA_MAX=input('Enter P_DELTA_MAX: ')
#                 if not P_DELTA_MAX.strip():
#                     raise ValueError('Field cannot be empty')
#                 P_DELTA_MAX=int(float(P_DELTA_MAX))
#                 if int(P_DELTA_MAX) < 0: 
#                     raise ValueError ('Field must greater than zero. Eg. 9999')
#             except ValueError as e:
#                 logger.debug(e)
#                 sys.exit('Exiting program...')

#             try:
#                 P_DELTA_MIN=input('Enter P_DELTA_MIN: ')
#                 if not P_DELTA_MIN.strip():
#                     raise ValueError('Field cannot be empty')
#                 P_DELTA_MIN=int(float(P_DELTA_MIN))
#                 if int(P_DELTA_MIN) > 0: 
#                     raise ValueError ('Field must greater than zero. Eg. -9999')
#             except ValueError as e:
#                 logger.debug(e)
#                 sys.exit('Exiting program...')

#             try:
#                 PROFIT=input('Enter PROFIT: ')
#                 MAXLOSS=input('Enter MAXLOSS: ')
#                 if not MAXLOSS.strip() or not PROFIT.strip():
#                     raise ValueError('Field cannot be empty')
#                 try:
#                     PROFIT=int(float(PROFIT))
#                     MAXLOSS=int(float(MAXLOSS))
#                 except ValueError as e: 
#                     logger.debug('Value must be integer or float') 
#                     sys.exit('Exiting program...')
#             except ValueError as e:
#                 logger.debug(e)
#                 sys.exit('Exiting program...')

#             try: 
#                 confirm = input('Enter 1 to proceed...\nEnter 2 to exit program... \n')
#                 if not confirm.strip(): 
#                     raise ValueError('Field cannot be empty')
#                 try: 
#                     confirm = int(float(confirm))
#                 except TypeError as e: 
#                     logger.debug('Integer or float values only')
#             except ValueError as e: 
#                 logger.debug(e)

#             if confirm == 1:
#                 st = Strategies(
#                 leg=leg, 
#                 sdate=sdate, 
#                 edate=edate, 
#                 dte=dte,
#                 user_cashflow=None,
#                 user_range=user_range,
#                 contract=contract, 
#                 delta=delta, 
#                 MID=MID,
#                 MID_STRIKE=MID_STRIKE,
#                 PC=PC, 
#                 strike=strike, 
#                 percent=percent, 
#                 distance=distance,
#                 PROFIT = PROFIT,
#                 MAXLOSS = MAXLOSS,
#                 P_DELTA_MIN = P_DELTA_MIN,
#                 P_DELTA_MAX = P_DELTA_MAX,
#                 INPUT_DAYS = INPUT_DAYS,
#                 strategy=strategy
#                 )
#                 st.run_conditions()
#             else:
#                 sys.exit('Program exited successfully...')

#         elif value == 2: 
#             pp = Preprocess()
#             pp.data_to_pickle()
#             files = glob.glob('./ProcessFiles/*.pkl')
#             pp.process_files(files)
#             self.run_calculators(self.filename)
            

        

