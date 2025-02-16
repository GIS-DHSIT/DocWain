from pymongo import MongoClient
import pandas as pd
from bson.objectid import ObjectId

from api.config import Config

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

mongoClient = MongoClient(Config.MongoDB.URI)
db = mongoClient[Config.MongoDB.DB]

dbName='documents'
docColl = db[dbName]
documents = docColl.find()
getProfiles = documents.collection.distinct('profile')
op = pd.DataFrame(docColl.find())
op[['name','profile','status']]
profiles = {}
for docs in documents:
    if docs['type'] == 'S3' and docs['profile'] in getProfiles:
        # if docs['status'] == 'TRAINING_PENDING':
        profName = docs['profile'].__str__()
        docName = docs['_id'].__str__()
        profiles[profName + '_' + docName] = docs
    # elif docs['type'] == 'LOCAL':
        # print(docs)
# for k,v in profiles.items():
conColl = db['connectors']
connectors = conColl.find()
conn = {}

for con in connectors:
    if con['name'] =='S3 test connection':
        print(con)
