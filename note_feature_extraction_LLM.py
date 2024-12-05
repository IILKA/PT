# Load model directly
from transformers import AutoTokenizer, AutoModel
import torch
import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

class TextDataset(Dataset):
    def __init__(self, texts, ids=None):
        self.texts = texts
        self.ids = ids if ids is not None else [i for i in range(len(texts))]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.ids[idx]

def get_notes_feature(texts, model_id="ProbeMedicalYonseiMAILab/medllama3-v20", batch_size=32, id_list=None, out_dir=None, gpu_id=0):
    '''
    texts: list of strings, 
    model_id: huggingface model id, 
    
    return or write 
    '''
    # Load model from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    if id_list is None:
        print("No id list provided, using index as id")

    dataset = TextDataset(texts, id_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    document_embeddings = []

    for batch_texts, batch_ids in tqdm(dataloader):
        #check if the file is already processed
        if out_dir:
            processed_files = os.listdir(out_dir)
            batch_ids = [id_ for id_ in batch_ids if f"{id_}.pt" not in processed_files]
            batch_texts = [text for id_, text in zip(batch_ids, batch_texts) if id_ in batch_ids]
            if len(batch_texts) == 0:       
                continue
        if isinstance(batch_texts, torch.Tensor):
            batch_texts = batch_texts.tolist()
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=False, truncation=False, max_length=None)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state
            batch_embedding = features.mean(dim=1)
        if out_dir:
            for idx, id_ in enumerate(batch_ids):
                torch.save(batch_embedding[idx], os.path.join(out_dir, f"{id_}.pt"))
        else:
            document_embeddings.append(batch_embedding.cpu())

    if not out_dir:
        document_embeddings = torch.cat(document_embeddings, dim=0)
        return document_embeddings

def worker(gpu_id, texts, model_id, batch_size, id_list, out_dir):
    get_notes_feature(texts, model_id=model_id, batch_size=batch_size, id_list=id_list, out_dir=out_dir, gpu_id=gpu_id)

if __name__ == "__main__":
    batch_size = 1

    #config here 
    #--- Set the number of GPUs to use and the output directory
    num_gpus = 8
    #--------------------------------------------------------
    
    out_dir = "./notes_embedding_ascelpius"
    df = pd.read_csv("./notes.csv")
    texts = df["text"].tolist()
    ids = df["id"].tolist()
    model_id = "starmpcc/Asclepius-Llama3-8B"
    
    num_texts_per_gpu = len(texts) // num_gpus

    processes = []
    # Split the texts into num_gpus parts and process them in parallel
    for i in range(num_gpus):
        start_idx = i * num_texts_per_gpu
        end_idx = (i + 1) * num_texts_per_gpu if i < num_gpus - 1 else len(texts)
        p = mp.Process(target=worker, args=(i, texts[start_idx:end_idx], model_id , batch_size, ids[start_idx:end_idx], out_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()