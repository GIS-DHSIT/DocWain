#!/usr/bin/env python3
"""Query MongoDB for profiles, documents, types, and statuses."""
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, 'src')

from api.config import Config
from pymongo import MongoClient

client = MongoClient(Config.MongoDB.URI)
db = client[Config.MongoDB.DB]

# List all collections
print("=== COLLECTIONS IN DATABASE ===")
for c in sorted(db.list_collection_names()):
    count = db[c].estimated_document_count()
    print(f"  {c}: ~{count} docs")
print()

# -------------------------------------------------------
# 1. All distinct profile_ids and subscription_ids
# -------------------------------------------------------
print("=" * 80)
print("1. DISTINCT PROFILES & SUBSCRIPTIONS")
print("=" * 80)

docs_col = db['documents']

pipeline_profiles = docs_col.aggregate([
    {"$group": {
        "_id": {
            "subscription_id": "$subscription_id",
            "profile_id": "$profile_id"
        },
        "doc_count": {"$sum": 1}
    }},
    {"$sort": {"_id.subscription_id": 1, "_id.profile_id": 1}}
])

profiles_map = {}
for rec in pipeline_profiles:
    sub_id = rec['_id'].get('subscription_id') or '(none)'
    prof_id = rec['_id'].get('profile_id') or '(none)'
    count = rec['doc_count']
    print(f"  subscription_id: {sub_id}")
    print(f"  profile_id:      {prof_id}")
    print(f"  document_count:  {count}")
    print()
    profiles_map[(sub_id, prof_id)] = count

print(f"Total unique (subscription, profile) pairs: {len(profiles_map)}")
print()

# -------------------------------------------------------
# 2. For each profile, list all documents with name, type, status
# -------------------------------------------------------
print("=" * 80)
print("2. DOCUMENTS PER PROFILE")
print("=" * 80)

for (sub_id, prof_id), _ in sorted(profiles_map.items()):
    query = {}
    if sub_id != '(none)':
        query['subscription_id'] = sub_id
    if prof_id != '(none)':
        query['profile_id'] = prof_id

    cursor = docs_col.find(query).sort('created_at', -1)

    print(f"\n--- subscription={sub_id} | profile={prof_id} ---")
    hdr = f"{'#':<3} {'Document Name':<55} {'Type/Domain':<20} {'Status':<25} {'Doc ID':<26}"
    print(hdr)
    print("-" * 130)

    for i, doc in enumerate(cursor, 1):
        name = (doc.get('document_name') or doc.get('name')
                or doc.get('filename') or doc.get('blob_name') or '(unknown)')
        if len(str(name)) > 52:
            name = str(name)[:49] + "..."
        doc_type = (doc.get('document_type') or doc.get('type')
                    or doc.get('category') or doc.get('domain') or '(none)')
        status = doc.get('status') or doc.get('processing_status') or '(none)'
        doc_id = str(doc.get('document_id') or doc.get('_id') or '')[:25]
        print(f"{i:<3} {str(name):<55} {str(doc_type):<20} {str(status):<25} {doc_id:<26}")

print()

# -------------------------------------------------------
# 3. Summary of document types/domains
# -------------------------------------------------------
print("=" * 80)
print("3. DOCUMENT TYPE / DOMAIN SUMMARY")
print("=" * 80)

type_agg = docs_col.aggregate([
    {"$group": {
        "_id": "$document_type",
        "count": {"$sum": 1},
        "statuses": {"$addToSet": "$status"},
        "sample_names": {"$push": "$document_name"}
    }},
    {"$sort": {"count": -1}}
])

for rec in type_agg:
    dtype = rec['_id'] or '(untyped)'
    count = rec['count']
    statuses = [s for s in (rec.get('statuses') or []) if s]
    samples = [n for n in (rec.get('sample_names') or []) if n][:5]
    print(f"\n  Type: {dtype}")
    print(f"  Count: {count}")
    print(f"  Statuses: {statuses}")
    print(f"  Sample docs: {samples}")

# Also check category field
print()
print("--- By category field ---")
cat_agg = docs_col.aggregate([
    {"$group": {"_id": "$category", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
])
for rec in cat_agg:
    cat = rec['_id'] or '(none)'
    print(f"  {cat}: {rec['count']}")

# Also check domain field
print()
print("--- By domain field ---")
dom_agg = docs_col.aggregate([
    {"$group": {"_id": "$domain", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
])
for rec in dom_agg:
    dom = rec['_id'] or '(none)'
    print(f"  {dom}: {rec['count']}")

# Distinct statuses
print()
print("--- All distinct statuses ---")
for s in docs_col.distinct('status'):
    print(f"  {s}")

# -------------------------------------------------------
# 4. Profiles collection
# -------------------------------------------------------
print()
print("=" * 80)
print("4. PROFILES COLLECTION")
print("=" * 80)
profiles_col = db['profiles']
for p in profiles_col.find().sort('created_at', -1):
    pid = p.get('profile_id') or p.get('_id')
    pname = p.get('profile_name') or p.get('name') or '(unnamed)'
    sub = p.get('subscription_id', '(none)')
    print(f"  profile_id: {pid}")
    print(f"  name: {pname}")
    print(f"  subscription_id: {sub}")
    for k in ['description', 'created_at', 'document_count', 'settings']:
        if k in p and p[k]:
            print(f"  {k}: {p[k]}")
    print()

# -------------------------------------------------------
# 5. Document field schema sample (first doc's keys)
# -------------------------------------------------------
print("=" * 80)
print("5. SAMPLE DOCUMENT SCHEMA (first doc's fields)")
print("=" * 80)
sample = docs_col.find_one()
if sample:
    for k, v in sorted(sample.items()):
        vtype = type(v).__name__
        vstr = str(v)
        if len(vstr) > 80:
            vstr = vstr[:77] + "..."
        print(f"  {k} ({vtype}): {vstr}")

client.close()
print("\nDone.")
