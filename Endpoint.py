#!/usr/bin/env python
# coding: utf-8

# In[10]:


from sagemaker.sklearn.estimator import SKLearn
from sagemaker import get_execution_role
import sagemaker
import os 

sagemaker_session = sagemaker.Session()
# region = sagemaker_session.boto_region_name
role = get_execution_role()

bucket_name = 'sagemaker-gacheon-003'
training_data_uri = os.path.join(f's3://{bucket_name}', 'data')
FRAMEWORK_VERSION = "0.20.0"
script_path = "train.py"

sklearn = SKLearn(
    entry_point=script_path,
    framework_version=FRAMEWORK_VERSION,
    instance_type="ml.c4.xlarge",
    role=role,
    sagemaker_session=sagemaker_session,
)


# In[11]:


get_ipython().system(' aws s3 ls s3://sagemaker-gacheon-003/data/')


# In[12]:


get_ipython().system('pygmentize train.py')


# In[13]:


conda install joblib


# In[14]:


# {"train": train_input}

sklearn.fit({"train":training_data_uri})


# In[ ]:


endpoint_name = 'ep-movie-rec-000003'
model_name = 'movie-recommend-model-000003'

predictor = sklearn.deploy(initial_instance_count=1, instance_type="ml.m5.xlarge",endpoint_name, model_name )


# In[ ]:


predictor.delete_endpoint()

