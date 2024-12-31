import pandas as pd 
from tqdm import tqdm

note_path = "./notes.csv" #this is the path to the notes.csv file
notes = pd.read_csv(note_path)
cnt = 0
#print first 1000 notes, id, text, and date
# for i in range(0, len(notes)):
#     print(notes.loc[i])
#     cnt += 1
#     if cnt == 50:
#         break

for i in tqdm(range(0, len(notes))):
    id = notes.loc[i, "id"]
    # with open(f"/mnt/data/home/ldy/mmiv_data/notes/{id}.txt", "w") as f:
    #     f.write(notes.loc[i, "text"])
    with open(f"/mnt/data/home/ldy/mmiv_data/notes_label/{id}.txt", "w") as f:
        f.write(notes.loc[i, ["subject_id", "hadm_id", "note_seq"]].to_string())
