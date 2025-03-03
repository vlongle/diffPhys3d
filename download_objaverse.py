import objaverse


uid=["f420ea9edb914e1b9b7adebbacecc7d8"]
objects = objaverse.load_objects(
uids=uid,
)
print("objects", objects)

# print(objaverse.load_annotations(uid)[uid[0]]['categories'])
