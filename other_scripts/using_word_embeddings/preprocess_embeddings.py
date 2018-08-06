import pandas as pd 
import numpy as np 
import re, string
from sklearn.metrics.pairwise import cosine_similarity as cos

EMBEDDING_FILE = "/home/weimin/yt8m/code/youtube-8m/wiki.en.vec"

def process(s):
    try:
        s = s.lower()
        s = re.sub("don't", "do not", s)
        s = re.sub("multi-tool", "multi tool", s)
        s = re.sub("tōshirō", "toshiro", s)
        s = re.sub("eight-string", "eight string", s)
        s = re.sub("Škoda", "skoda", s)
        s = re.sub("abadá", "abadá", s)
        s = re.sub("citroën", "citroen", s)
        s = re.sub("pokémon", "pokemon", s)
        s = re.sub("middleearth", "middle earth", s)

        s = re.sub("mercedes-benz", "mercedes benz", s)

        s = re.sub("turbografx-16", "turbografx", s)
        s = re.sub("è", "e", s)
        s = re.sub(r"(\w+)'(\w* )", r"\1 ", s) # Carl Jones's -> Carl Jones
        s = re.sub(r"[^a-zA-Z\d\s]", "", s) # remove all non-alphanumeric characters 
        s = re.sub(" +", " ", s)
        s = s.strip()
        s = re.sub("mazdaspeed3", "mazda speed", s)
        s = re.sub("scoobydoo", "scooby doo", s)

        s = re.sub("mazda3", "mazda", s)
        s = re.sub("mazda6", "mazda", s)
        s = re.sub("metin2", "video game", s)

        s = re.sub("beamngdrive", "video game", s)
        s = re.sub("dmcgh4", "panasonic lumix camera", s)
        s = re.sub("reventn", "Lamborghini", s) 
        s = re.sub("ibnlaahad", "video game", s) 
        s = re.sub("steelstring", "steel string", s) 
        s = re.sub("drivethrough", "drive through", s) 
        s = re.sub("richtertuned", "richter tuning", s) 
        s = re.sub("clsclass", "mercedes benz", s) 
        s = re.sub("thousandyear", "thousand year", s) 
        s = re.sub("koriandr", "parsley", s) 
        s = re.sub("play4free", "video game", s) 
        s = re.sub("slkclass", "mercedes benz", s) 
        s = re.sub("handtohand", "hand to hand", s) 
        s = re.sub("lowcarbohydrate", "low carbohydrate", s) 
        s = re.sub("presequel", "video game", s) 
        s = re.sub("amazoncom", "website", s) 
        s = re.sub("dgrayman", "manga", s) 
        s = re.sub("fixedgear", "bicycle", s) 
        s = re.sub("gshock", "watch", s) 
        s = re.sub("citybuilding", "city building", s) 
        s = re.sub("twowheel", "two wheel", s) 
        s = re.sub("twostroke", "two stroke", s) 
        s = re.sub("radiocontrolled", "radio controll", s) 
        s = re.sub("mercedesamg", "mercedes", s) 
        s = re.sub("singlelens", "single lens", s) 
        s = re.sub("afrotextured", "afro texture", s) 
        s = re.sub("actionadventure", "action adventure", s) 
        s = re.sub("fourwheel", "four wheel", s) 
        s = re.sub("djinshi", "japanese manga", s) 
    except:
        print("\n\nCannot process this: , ", s)

    return s 

def process_vertical(s):
    try:
        s = re.sub("Games", "game", s)
        s = re.sub("Arts & Entertainment", "art entertainment", s)

        s = re.sub("Autos & Vehicles", "auto vehicle", s)
        s = re.sub("Computers & Electronics", "computer electronics", s)
        s = re.sub("Food & Drink", "food drink", s)
        s = re.sub("Business & Industrial", "business industrial", s)
        s = re.sub("Sports", "sports", s)
        s = re.sub("Pets & Animals", "pet animal", s)
        s = re.sub("\\(Unknown\\)", "unknown", s)
        s = re.sub("Hobbies & Leisure", "hobby leisure", s)
        s = re.sub("Home & Garden", "home garden", s)
        s = re.sub("Science", "science", s)
        s = re.sub("Shopping", "shopping", s)
        s = re.sub("Beauty & Fitness", "beauty fitness", s)
        s = re.sub("Travel", "travel", s)
        s = re.sub("People & Society", "people society", s)
        s = re.sub("Reference", "reference", s)
        s = re.sub("Law & Government", "law government", s)
        s = re.sub("Internet & Telecom", "internet telecom", s)
        s = re.sub("Books & Literature", "book literature", s)
        s = re.sub("News", "news", s)
        s = re.sub("Health", "health", s)
        s = re.sub("Jobs & Education", "job education", s)
        s = re.sub("Finance", "finance", s)
        s = re.sub("Real Estate", "real estate", s)

    except: 
        return ""
    return s

df = pd.read_csv('./vocabulary.csv')

df['Vertical1'] = df['Vertical1'].apply(process_vertical)
df['Vertical2'] = df['Vertical2'].apply(process_vertical)
df['Vertical3'] = df['Vertical3'].apply(process_vertical)

df = df[['Index', 'Name', 'WikiDescription', 'Vertical1', 'Vertical2', 'Vertical3']]

df.fillna('unknown', inplace = True)

df['Name'] = df['Name'].apply(process)
df['WikiDescription'] = df['WikiDescription'].apply(process)

names_str = df['Name'].values
desc_str = df['WikiDescription'].values
verticals = df['Vertical1'].values + ' ' + df['Vertical2'].values + ' ' + df['Vertical3'].values 
for i in range(len(verticals)):
    verticals[i] = verticals[i].strip()

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

### stats 
row_count = 0 
row_miss = []
words_count = 0 
words_miss = [] 
for r in names_str:
    missed = True
    row_count += 1
    for w in r.split():
        words_count += 1
        if w in embeddings_index:
            missed = False 
        else:
            words_miss.append(w)
    if missed:
        row_miss.append(r)

print("Total missed rows: ", len(row_miss))
print("Total missed words: ", len(words_miss))
print("Total words: ", words_count)

name_vec = np.concatenate([np.mean([embeddings_index.get(w, np.zeros(300)).reshape(1, 300) for w in r.split()], 0) for r in names_str], axis = 0)
desc_vec = np.concatenate([np.mean([embeddings_index.get(w, np.zeros(300)).reshape(1, 300) for w in r.split()], 0) for r in desc_str], axis = 0)
vertical_vec = np.concatenate([np.mean([embeddings_index.get(w, np.zeros(300)).reshape(1, 300) for w in r.split()], 0) for r in verticals], axis = 0)

final_vec = (name_vec + desc_vec + vertical_vec) / 3
final_vec_mean = final_vec.mean(0).reshape(1, 300)
final_vec_std = final_vec.std(0).reshape(1, 300)
final_vec = (final_vec - final_vec_mean) / final_vec_std

final_vec_df = pd.DataFrame(final_vec)
final_vec_df['Index'] = df['Index'].values

final_vec_df.sort_values('Index', inplace = True)

final_vec_df.to_csv('./vec_ready_to_use_v1.0.csv', index = False)

dfl = df.copy()

dfl.sort_values('Index', inplace = True)

### Testing 
testing_label_id = 0

mat = final_vec_df.iloc[:, :300].values 
game_idx = np.where(final_vec_df['Index'] == testing_label_id)[0][0]
res = cos(mat[game_idx], mat)
top_10_idx = res[0].argsort()[-10:][::-1]
top_10_label_ids = final_vec_df['Index'].values[top_10_idx]

print(dfl.loc[dfl['Index'].isin(top_10_label_ids), :])







