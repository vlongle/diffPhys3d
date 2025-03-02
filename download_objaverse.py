import objaverse


uid=["ecb91f433f144a7798724890f0528b23"]
objects = objaverse.load_objects(
uids=uid,
)
print("objects", objects)
