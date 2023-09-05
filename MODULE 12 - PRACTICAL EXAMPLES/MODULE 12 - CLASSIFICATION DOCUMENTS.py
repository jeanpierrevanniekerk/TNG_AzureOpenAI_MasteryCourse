# Databricks notebook source
# MAGIC %md
# MAGIC # Classify Documents

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC
# MAGIC from IPython.core.interactiveshell import InteractiveShell
# MAGIC InteractiveShell.ast_node_interactivity = "all"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# MAGIC %%capture
# MAGIC !pip install -r ../requirements.txt

# COMMAND ----------

import pandas as pd
df = spark.sql('select * from openai.document_analysis_embeddings').toPandas()

# COMMAND ----------

# drop rows with NaN
df.dropna(inplace=True)
len(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classify documents with their embeddings
# MAGIC ref: https://github.com/openai/openai-cookbook/blob/main/examples/Classification_using_embeddings.ipynb

# COMMAND ----------

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# COMMAND ----------

X = df['embedding']

map_dict = {"business": 0, "entertainment":1, "politics":2, "sport":3, "tech":4}
df['category'].replace(map_dict, inplace=True)

y = df['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True, stratify=y) 

# reshape X into 2D array
X_train = np.vstack(X_train)
X_test = np.vstack(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit a model with `XGBoost`

# COMMAND ----------

from xgboost import XGBClassifier
import pickle

TRAIN = True
LOAD = False

# filename for trained model
fname = '../output/models/xgb.pkl'

if TRAIN: 
    # create model instance
    xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=1, objective='multi:softprob')
    # fit model
    xgb.fit(X_train, y_train)

# predict
preds = xgb.predict(X_test)
probas = xgb.predict_proba(X_test)

# report
report = classification_report(y_test, preds)
print(report)

# COMMAND ----------

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# confusion matrix
cm = confusion_matrix(y_test, preds)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=xgb.classes_).plot()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit a model with `RandomForest`

# COMMAND ----------

TRAIN = True
LOAD = False
# filename for trained model
fname = '../output/models/rf.pkl'

# fit model
if TRAIN: 
    # train random forest classifier
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)

# predict
rf_preds = rf.predict(X_test)
rf_probas = rf.predict_proba(X_test)

# report
rf_report = classification_report(y_test, rf_preds)
print(rf_report)

# COMMAND ----------

# Confusion matrix
rf_cm = confusion_matrix(y_test, rf_preds)
rf_cm_display = ConfusionMatrixDisplay(confusion_matrix=rf_cm, display_labels=rf.classes_).plot()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save prediction

# COMMAND ----------

y_test_pred = pd.DataFrame()
y_test_pred['category'] = y_test
y_test_pred['prediction'] = preds
y_test_pred

# COMMAND ----------

df_test_result = pd.concat([df, y_test_pred.drop(columns='category')], axis=1, join="inner")
# df_test_result.shape
df_test_result1 = spark.createDataFrame(df_test_result)
df_test_result1.write.mode('overwrite').option("overwriteSchema", "true").saveAsTable('openai.document_analysis_predictions')
df_test_result1.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wrong Predictions

# COMMAND ----------

import pyspark.sql.functions as psf
df_wrong_predictions = spark.sql('select * from openai.document_analysis_predictions').where(psf.col('category') != psf.col('prediction'))
df_wrong_predictions.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("openai.document_analysis_wrong_predictions")
df_wrong_predictions.toPandas()

# COMMAND ----------


