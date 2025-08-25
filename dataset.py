import kagglehub
import shutil
import os
import pandas as pd

local_path = os.path.abspath('./data-set')
def download():
	if(os.path.exists(local_path)):
		return

	dataset_path = kagglehub.dataset_download("blastchar/telco-customer-churn")

	shutil.move(dataset_path, local_path)
	
def import_as_df():
	return pd.read_csv(f'{local_path}/WA_Fn-UseC_-Telco-Customer-Churn.csv')