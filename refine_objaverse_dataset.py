import objaverse
from sentence_transformers import SentenceTransformer, util
import torch
import pickle 

uid = objaverse.load_uids()

annotations = objaverse.load_annotations(uid)

category_dict = {
  "cloth_and_fabrics": [
    "apron", "ballet_skirt", "bandanna", "bedspread", "belt", "beret", "blanket", 
    "blouse", "boiled_egg", "bolo_tie", "bonnet", "bow-tie", "bow_(decorative_ribbons)", 
    "brassiere", "breechcloth", "bridal_gown", "cap_(headwear)", "cape", "cardigan", 
    "carpet", "cincture", "cloak", "corset", "costume", "crown", "curtain", "diaper", 
    "dress", "dress_hat", "dress_suit", "flag", "fleece", "handkerchief", "hat", 
    "headscarf", "jacket", "jean", "jersey", "kimono", "lab_coat", "legging_(clothing)", 
    "napkin", "neckerchief", "necktie", "pajamas", "pillow", "poncho", "quilt", 
    "raincoat", "robe", "scarf", "shawl", "shirt", "skirt", "sock", "swimsuit", 
    "tablecloth", "tapestry", "tie", "tights_(clothing)", "towel", "trousers", 
    "turtleneck_(clothing)", "umbrella", "underdrawers", "underwear", "veil", "vest"
  ],
  "rigid_containers": [
    "aquarium", "backpack", "barrel", "basket", "bathtub", "bottle", "bowl", 
    "box", "bucket", "cabinet", "can", "canister", "carton", "chest", "cistern", 
    "cup", "drawer", "dumpster", "envelope", "fishbowl", "flask", "glass_(drink_container)", 
    "jar", "jug", "mailbox", "mug", "pitcher_(vessel_for_liquid)", "pot", "saucepan", 
    "suitcase", "tank_(storage_vessel)", "thermos_bottle", "trash_can", "tray", 
    "urn", "vase", "water_bottle", "water_cooler", "water_jug"
  ],
  "deformable_organics": [
    "alligator", "apple", "apricot", "artichoke", "asparagus", "avocado", "banana", 
    "bear", "beef_(food)", "bell_pepper", "bird", "blackberry", "blueberry", 
    "broccoli", "brussels_sprouts", "calf", "camel", "cantaloup", "carrot", 
    "cat", "cauliflower", "celery", "cherry", "chicken_(animal)", "clementine", 
    "cow", "crab_(animal)", "cucumber", "date_(fruit)", "deer", "dog", "dolphin", 
    "duck", "eagle", "egg", "elephant", "fig_(fruit)", "fish", "flamingo", 
    "flower_arrangement", "frog", "giraffe", "goat", "gorilla", "grape", "gull", 
    "hamster", "heron", "hippopotamus", "horse", "kitten", "kiwi_fruit", "koala", 
    "lamb_(animal)", "lemon", "lettuce", "lime", "lion", "lizard", "mandarin_orange", 
    "melon", "monkey", "mushroom", "octopus_(animal)", "onion", "orange_(fruit)", 
    "ostrich", "papaya", "peach", "pear", "penguin", "persimmon", "pineapple", 
    "potato", "pumpkin", "rabbit", "radish", "raspberry", "rat", "sheep", 
    "snake", "squirrel", "strawberry", "sweet_potato", "tiger", "tomato", 
    "turtle", "watermelon", "zebra", "zucchini"
  ],
  "semi_rigid_structures": [
    "armchair", "bed", "bench", "bicycle", "book", "bookcase", "box", "cabinet", 
    "chair", "chaise_longue", "Christmas_tree", "clipboard", "coatrack", "corkboard", 
    "couch", "desk", "door", "doormat", "easel", "filing_cabinet", "furniture", 
    "hammock", "ladder", "lamp", "loveseat", "mattress", "ottoman", "picture_frame", 
    "recliner", "rocking_chair", "shelving", "shovel", "sofa", "sofa_bed", "stool", 
    "table", "trampoline", "wheelchair", "wooden_spoon"
  ],
  "rigid_tools": [
    "ax", "baseball_bat", "bolt", "broom", "camera", "can_opener", "candle", "cane", 
    "chisel", "chopstick", "clippers_(for_plants)", "corkscrew", "crowbar", "crutch", 
    "dagger", "flashlight", "fork", "frying_pan", "hammer", "handsaw", "hoe", 
    "hose", "knife", "ladder", "ladle", "mallet", "oar", "pan_(for_cooking)", 
    "pencil", "pen", "pliers", "rake", "razor", "ruler", "saw", "scissors", 
    "screwdriver", "shovel", "spatula", "spear", "spoon", "stapler", "sword", 
    "tennis_racket", "tongs", "toothbrush", "umbrella", "wagon", "walking_cane", 
    "walking_stick", "wrench"
  ],
  "mechanical_objects": [
    "airplane", "bicycle", "blender", "calculator", "camera", "car_(automobile)", 
    "cassette", "clock", "computer_keyboard", "doorknob", "drone", "eggbeater", 
    "fan", "food_processor", "golfcart", "gun", "hair_dryer", "helicopter", "iPod", 
    "iron_(for_clothing)", "laptop_computer", "lawn_mower", "lock", "microwave_oven", 
    "mixer_(kitchen_tool)", "motorcycle", "pencil_sharpener", "printer", "refrigerator", 
    "remote_control", "router_(computer_equipment)", "sewing_machine", "shaver_(electric)", 
    "telephone", "television_set", "toaster", "tractor_(farm_equipment)", "typewriter", 
    "vacuum_cleaner", "washing_machine"
  ],
  "fluid_containers": [
    "beer_bottle", "beer_can", "bottle", "Dixie_cup", "flower_vase", "glass_(drink_container)", 
    "kettle", "mug", "perfume", "pop_(soda)", "teacup", "teakettle", "teapot", 
    "thermos_bottle", "vase", "water_bottle", "water_jug", "wineglass"
  ],
  "soft_deformable": [
    "balloon", "beanbag", "cushion", "doll", "puppet", "rag_doll", "rubber_band", 
    "stuffed_animal", "teddy_bear", "toy"
  ],
  "elastic_materials": [
    "balloon", "elastic_band", "rubber_band", "rubber_glove", "swimsuit", 
    "tights_(clothing)", "trampoline"
  ],
  "tree": [
      "tree", "ficus", "fern"
  ]
}



model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

# Get object IDs and their names
all_obj_ids = list(annotations.keys())
all_obj_names = [annotations[obj_id].get("name", "") for obj_id in all_obj_ids]

# Encode all object descriptions using a specified batch size
obj_embeddings = model.encode(
    all_obj_names,
    batch_size=128,   
    convert_to_tensor=True, 
    show_progress_bar=True   
)

print("Embeddings shape:", obj_embeddings.shape)

model = SentenceTransformer("all-MiniLM-L6-v2", device='cuda')

# Dictionary: object_id -> (category, similarity_score)
from collections import defaultdict
assignment = {}

top_k = 500

for cat_key, cat_list in category_dict.items():
    # 1. Create the query text by combining subcategories
    cat_query = " ".join(cat_list)
    cat_embedding = model.encode(cat_query, convert_to_tensor=True)
    
    # 2. Similarities for all objects (shape: [N_objects])
    similarities = util.cos_sim(cat_embedding, obj_embeddings)[0]
    
    # 3. Get the top-k indices and their similarity values
    topk = torch.topk(similarities, k=top_k)
    topk_indices = topk.indices
    topk_values  = topk.values
    
    # 4. For each top-k object, see if it should be assigned (or reassigned)
    for i, obj_idx in enumerate(topk_indices):
        obj_id = all_obj_ids[obj_idx]
        sim_score = topk_values[i].item()
        
        # Check if this object hasn't been assigned yet OR 
        # if this category has a higher similarity than the previous assignment
        if (obj_id not in assignment) or (sim_score > assignment[obj_id][1]):
            assignment[obj_id] = (cat_key, sim_score)

# Build final_dataset = { category: [(object_id, sim_score)...] }
final_dataset = defaultdict(list)
for obj_id, (cat_key, sim_score) in assignment.items():
    final_dataset[cat_key].append((obj_id, sim_score))

final_dataset = dict(final_dataset)

# final_dataset
with open('final_dataset.pkl', 'wb') as f:
    pickle.dump(final_dataset, f)
