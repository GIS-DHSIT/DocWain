
from pymongo import MongoClient
import pandas as pd
from bson.objectid import ObjectId

from src.api.config import Config

# Connect to MongoDB
mongoClient = MongoClient(Config.MongoDB.URI)
db = mongoClient[Config.MongoDB.DB]

# Documents collection
docColl = db['documents']
documents = list(docColl.find())  # convert to list for re-use

# Distinct profiles
getProfiles = docColl.distinct('profile')

# DataFrame view
op = pd.DataFrame(documents)
print(op[['name', 'profile', 'status']])  # ✅ now shows output

# Build profiles dictionary
profiles = {}
for docs in documents:
    if docs['type'] == 'S3' and docs['profile'] in getProfiles:
        profName = str(docs['profile'])
        docName = str(docs['_id'])
        profiles[f"{profName}_{docName}"] = docs

# Connectors collection
conColl = db['connectors']
connectors = list(conColl.find())

for con in connectors:
    if con['name'] == 'S3 test connection':
        print(con)
