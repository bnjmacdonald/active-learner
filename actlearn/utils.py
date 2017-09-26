import datetime
from bson import ObjectId

def datetime_objectid_handler(x):
    if isinstance(x, datetime.datetime):
        return x.isoformat()
    elif isinstance(x, ObjectId):
            return str(x)
    raise TypeError("Unknown type")
