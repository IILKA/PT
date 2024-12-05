import pickle
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import copy
import time

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from Data import EHRDataset, collate_fn
from model import TransformerClassifier, PositionalEncoding, AttentionPooling, LabelSmoothingFocalLoss




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_EHR(ehr_path = "ehr_preprocessed_seq_by_day_cat_embedding.pkl", ensemble = False ):

    #data preprocessing
    with open(ehr_path, 'rb') as f:
        data = pickle.load(f)
        feat_dict = data['feat_dict']

    # load datasets
    train_df = pd.read_csv('train.csv')
    valid_df = pd.read_csv('valid.csv')


    # add valid to train
    if ensemble:
        train_df = pd.concat([train_df, valid_df], ignore_index=True)
    test_df = pd.read_csv('test.csv')

    # handling the conflicting labels in train set
    train_labels_df = train_df[['id', 'readmitted_within_30days']].drop_duplicates()
    label_counts_train = train_labels_df.groupby('id')['readmitted_within_30days'].nunique()
    conflicting_ids_train = label_counts_train[label_counts_train > 1].index.tolist()

    if len(conflicting_ids_train) > 0:
        print(f"Found IDs with conflicting labels in train set: {conflicting_ids_train}")
        train_labels_df = train_labels_df[~train_labels_df['id'].isin(conflicting_ids_train)]
    else:
        print("No conflicting labels found for IDs in train set.")

    #handling the conflicting labels in valid set
    valid_labels_df = valid_df[['id', 'readmitted_within_30days']].drop_duplicates()
    label_counts_valid = valid_labels_df.groupby('id')['readmitted_within_30days'].nunique()
    conflicting_ids_valid = label_counts_valid[label_counts_valid > 1].index.tolist()


    test_labels_df = test_df[['id']].drop_duplicates()

    if len(conflicting_ids_valid) > 0:
        print(f"Found IDs with conflicting labels in validation set: {conflicting_ids_valid}")
        valid_labels_df = valid_labels_df[~valid_labels_df['id'].isin(conflicting_ids_valid)]
    else:
        print("No conflicting labels found for IDs in validation set.")

    # 定义 train_ids, train_labels, valid_ids, valid_labels, test_ids
    train_ids = train_labels_df['id'].tolist()
    train_labels = train_labels_df['readmitted_within_30days'].tolist()

    valid_ids = valid_labels_df['id'].tolist()
    valid_labels = valid_labels_df['readmitted_within_30days'].tolist()

    test_ids = test_labels_df ['id'].tolist()

    # 2.feature selection using RandomForest
    # ----------------------------

    print("Performing feature selection using RandomForest...")
    X_train_rf = np.vstack([feat_dict[id_].mean(axis=0) for id_ in train_ids if id_ in feat_dict])
    y_train_rf = np.array([label for id_, label in zip(train_ids, train_labels) if id_ in feat_dict])

    # init rf
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_rf, y_train_rf)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    selected_features = indices[:100]  #select the most important 100 features

    # update feat_dict，only keep the selected features
    for id_ in feat_dict:
        feat_matrix = feat_dict[id_]
        if feat_matrix.shape[1] >= 100:
            feat_dict[id_] = feat_matrix[:, selected_features]
        else:
            feat_dict[id_] = np.zeros((feat_matrix.shape[0], 100))  # replace with zeros 


    #3. Standard Scaler
    #-------------------------
    print("Applying StandardScaler to the data...")
    all_train_features = np.vstack([feat_dict[id_] for id_ in train_ids if id_ in feat_dict])
    scaler = StandardScaler()
    scaler.fit(all_train_features)

    # calculate mean
    mean_feature = all_train_features.mean(axis=0)

    # standardize features
    for dataset_ids in [train_ids, valid_ids, test_ids]:
        for id_ in dataset_ids:
            if id_ in feat_dict:
                feat_dict[id_] = scaler.transform(feat_dict[id_])
            else:
                #replace missing IDs with zeros 
                feat_dict[id_] = np.zeros((feat_dict[list(feat_dict.keys())[0]].shape[1], 100))
            # feat_dict[id_] = #mean_feature  # replace missing IDs with mean feature
    return feat_dict, train_ids, train_labels, valid_ids, valid_labels, test_ids, mean_feature
    



def handle_ensemble(feat_dict, train_ids, train_labels, valid_ids, valid_labels, test_ids, mean_feature, ensemble = False, result_dir = "./results", test_name = "test_result"):
    print("training process starts")
    #----------------------- batch size is defined here -----------------------
    batch_size = 128  
    if ensemble:
        K = 10
        kf = KFold(n_splits=K, shuffle=True, random_state=42)
        for fold, (train_index, valid_index) in enumerate(kf.split(train_ids)):
            test_name = f"{test_name}_{K}>k={fold}"
            print(f"Fold {fold+1}:")
            train_ids_fold = [train_ids[i] for i in train_index]
            valid_ids_fold = [train_ids[i] for i in valid_index]

            train_labels_fold = [train_labels[i] for i in train_index]
            valid_labels_fold = [train_labels[i] for i in valid_index]
              
            train_dataset = EHRDataset(feat_dict, train_ids_fold, train_labels_fold, mean_feature=mean_feature, noise_std=0.05, device=device)
            valid_dataset = EHRDataset(feat_dict, valid_ids_fold, valid_labels_fold, mean_feature=mean_feature, device=device)
            
          

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) 
            
            test_dataset = EHRDataset(feat_dict, test_ids, mean_feature=mean_feature, device=device)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            
            trainer(train_loader, valid_loader, test_name, result_dir=result_dir)
        return train_loader, valid_loader, test_loader
    else:
        train_dataset = EHRDataset(feat_dict, train_ids, train_labels, mean_feature=mean_feature, noise_std=0.05, device=device)
        valid_dataset = EHRDataset(feat_dict, valid_ids, valid_labels, mean_feature=mean_feature, device=device)
        test_dataset = EHRDataset(feat_dict, test_ids, mean_feature=mean_feature, device=device)

        batch_size = 128

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        return train_loader, valid_loader, test_loader

def create_test_loader(feat_dict, test_ids, mean_feature, device = device):
    test_dataset = EHRDataset(feat_dict, test_ids, mean_feature=mean_feature, device=device)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    return test_loader



def trainer(train_loader, valid_loader, test_name, device = device, result_dir = "./results"):
    #running test: 
    print(f"Running test: {test_name}")
    print(f"Using device: {device}")
    model = TransformerClassifier().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = LabelSmoothingFocalLoss(alpha=class_weights[1].item(), gamma=2, smoothing=0.1, reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-1)  # 降低学习率和权重衰减

    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(epoch):
        warm_up_epochs = 5
        if epoch < warm_up_epochs:
            return float(epoch + 1) / float(warm_up_epochs)
        else:
            return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=5e-4, T_max=100)

    scaler = GradScaler()

    def initialize_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(initialize_weights)

    #----------------------- Define training params here -----------------------
    num_epochs = 150
    best_val_auc = 0.0
    best_val_loss = float('inf')
    best_model_state = copy.deepcopy(model.state_dict())
    patience = 20
    counter = 0


    for epoch in tqdm(range(num_epochs), desc="Epochs", leave=True):
        model.train()
        total_loss = 0
        train_labels_list = []
        train_probs_list = []

        for batch in train_loader:
            sequences, notes, src_key_padding_mask, src_key_padding_mask_notes, labels, _ = batch
            # sequences, notes, src_key_padding_mask, labels, _ = batch
            sequences = sequences.to(device)
            notes = notes.to(device)
            src_key_padding_mask = src_key_padding_mask.to(device)
            labels = labels.to(device)
            src_key_padding_mask_notes = src_key_padding_mask_notes.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(sequences, notes, src_key_padding_mask, src_key_padding_mask_notes)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            # outputs = torch.sigmoid(outputs)
            train_probs_list.extend(outputs.detach().cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        try:
            train_auc = roc_auc_score(train_labels_list, train_probs_list)
        except ValueError:
            train_auc = 0.0
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, AUC: {train_auc:.4f}")

        # 验证
        model.eval()
        val_loss = 0
        val_labels_list = []
        val_probs_list = []
        with torch.no_grad():
            for batch in valid_loader:
                sequences, notes, src_key_padding_mask, src_key_padding_mask_notes ,labels, _ = batch
                sequences = sequences.to(device)
                notes = notes.to(device)
                src_key_padding_mask = src_key_padding_mask.to(device)
                src_key_padding_mask_notes = src_key_padding_mask_notes.to(device)
                labels = labels.to(device)

                with autocast():
                    outputs = model(sequences, notes, src_key_padding_mask, src_key_padding_mask_notes)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                # outputs = torch.sigmoid(outputs)
                val_probs_list.extend(outputs.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        val_loss /= len(valid_loader)
        try:
            val_auc = roc_auc_score(val_labels_list, val_probs_list)
        except ValueError:
            val_auc = 0.0

        print(f"Validation Loss: {val_loss:.4f}, AUC: {val_auc:.4f}\n")

        # 早停
        if val_auc > best_val_auc: # or (val_auc > 0.77 and train_auc > 0.8):
            best_val_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())
            counter = 0
            print('  Saved Best Model!')
        else:
            counter += 1
            if counter >= patience:
                print('Early stopping triggered!')
                print("the best validation AUC is: ", best_val_auc)
                print("the model parameter number is: ", total_params)
                break

        scheduler.step()

    model.load_state_dict(best_model_state)

    #save the model 
    model_out_path = os.path.join(result_dir, f'{test_name}.pth')
    torch.save(model.state_dict(), model_out_path)
    return best_model_state, model_out_path

def test(model, test_loader, test_ids, test_name, result_dir = "./results"):
   

    model.eval()
    test_predictions = {id_: None for id_ in test_ids}  # Use a dictionary to preserve order

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            sequences, notes, src_key_padding_mask, src_key_padding_mask_notes, ids = batch
            sequences = sequences.to(device)
            notes = notes.to(device)
            src_key_padding_mask = src_key_padding_mask.to(device)
            src_key_padding_mask_notes = src_key_padding_mask_notes.to(device)

            with autocast():
                outputs = model(sequences, notes, src_key_padding_mask, src_key_padding_mask_notes)
            # outputs = torch.sigmoid(outputs).cpu().numpy()
            outputs = outputs.cpu().numpy()
            for id_, prob in zip(ids, outputs):
                test_predictions[id_] = prob


    # Ensure all IDs have predictions
    assert None not in test_predictions.values(), "Some IDs are missing predictions."

    # Output predictions to CSV
    test_results_df = pd.DataFrame({
        'id': test_ids,
        'readmitted_within_30days': [test_predictions[id_] for id_ in test_ids]
    })

    out_path = os.path.join(result_dir, f'{test_name}.csv')
    test_results_df.to_csv(out_path, index=False)

    print(f"Testing completed. Predictions saved to '{test_name}.csv'.")


if __name__ == '__main__':
    import argparse
    
    time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="train+test", help="train, test, train+test")
    parser.add_argument("--ensemble", type=bool, default=False, help="ensemble training")
    parser.add_argument("--model_path", type=str, default=None, help="model path")
    parser.add_argument("--result_dir", type=str, default=f"./results/{time}", help="result directory")

    
    args = parser.parse_args()
    task = args.task
    is_ensemble = args.ensemble
    result_dir = args.result_dir
    feat_dict, train_ids, train_labels, valid_ids, valid_labels, test_ids, mean_feature = prepare_EHR(ensemble=is_ensemble)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if task == "train":
        train_loader, valid_loader, test_loader = handle_ensemble(
            feat_dict,
            train_ids, 
            train_labels,
            valid_ids, 
            valid_labels, 
            test_ids, 
            mean_feature, 
            ensemble = is_ensemble, 
            result_dir = result_dir
        )
        if is_ensemble: 
            print("Ensemble training process finished")
        else:
             best_model_state, best_model_path = trainer(train_loader, valid_loader, "best_model_test", result_dir = result_dir)
    if  task == "train+test":
        train_loader, valid_loader, test_loader = handle_ensemble(
            feat_dict,
            train_ids, 
            train_labels,
            valid_ids, 
            valid_labels, 
            test_ids, 
            mean_feature, 
            ensemble = is_ensemble, 
            result_dir = result_dir
        )
        if is_ensemble: 
            model_name_list = os.listdir(result_dir)
            model_name_list = [model_name for model_name in model_name_list if model_name.endswith(".pth")]
            for model_name in model_name_list:
               model_path = os.path.join(result_dir, model_name)
               model = TransformerClassifier().to(device)
               model.load_state_dict(torch.load(model_path))
               test(model, test_loader, test_ids, test_name = model_name.split(".")[0],result_dir = result_dir)
            
            files = os.listdir(result_dir)
            print("files:", files)
            files = [f for f in files if f.endswith(".csv")]
            df = pd.read_csv(os.path.join(result_dir, files[0]))

            # Save the 'id' column separately to keep it intact
            id_column = df['id']

            # Select only the numeric columns (excluding 'id')
            df_numeric = df.select_dtypes(include=[np.number])
            for csv_file in files[1:]:
                # Read the current CSV file
                temp_df = pd.read_csv(os.path.join(result_dir, csv_file))
                
                # Save the 'id' column separately from the rest
                temp_id_column = temp_df['id']
                
                # Select only the numeric columns (excluding 'id')
                temp_df_numeric = temp_df.select_dtypes(include=[np.number])

                # Add the numeric columns to the main DataFrame
                df_numeric = df_numeric.add(temp_df_numeric, fill_value=0)

            # Calculate the average of the numeric columns
            df_numeric = df_numeric / len(files)
            print(len(files))

            # Combine the 'id' column back with the averaged numeric columns
            final_df = pd.concat([id_column, df_numeric], axis=1)

            # Save the result to a new CSV file
            out_path = os.path.join(result_dir, f'ensemble_result.csv')
            print("the ensemble result is saved to ", out_path)
            final_df.to_csv(out_path, index=False)
        else:
            best_model_state, best_model_path = trainer(train_loader, valid_loader, "best_model_test", result_dir = result_dir)
            model = TransformerClassifier().to(device)
            model.load_state_dict(torch.load(best_model_path))
            test(model, test_loader, test_ids, test_name = "best_model_test", result_dir = result_dir)
    if task == "test":
        test_loader = create_test_loader(feat_dict, test_ids, mean_feature, device = device)
        if is_ensemble:
            model_name_list = os.listdir(result_dir)
            model_name_list = [model_name for model_name in model_name_list if model_name.endswith(".pth")]
            for model_name in model_name_list:
                model_path = os.path.join(result_dir, model_name)
                model = TransformerClassifier().to(device)
                model.load_state_dict(torch.load(model_path))
                test(model, test_loader, test_ids, test_name = model_name.split(".")[0], result_dir = result_dir)
            files = os.listdir(result_dir)
            files = [f for f in files if f.endswith(".csv")]
            files = [f for f in files if "ensemble" not in f]
            print("calculating files:", files)
            print(files[0])
            df = pd.read_csv(os.path.join(result_dir, files[0]))

            # Save the 'id' column separately to keep it intact
            id_column = df['id']

            # Select only the numeric columns (excluding 'id')
            df_numeric = df.select_dtypes(include=[np.number])
            for csv_file in files[1:]:
                # Read the current CSV file
                temp_df = pd.read_csv(os.path.join(result_dir, csv_file))
                
                # Save the 'id' column separately from the rest
                temp_id_column = temp_df['id']
                
                # Select only the numeric columns (excluding 'id')
                temp_df_numeric = temp_df.select_dtypes(include=[np.number])

                # Add the numeric columns to the main DataFrame
                df_numeric = df_numeric.add(temp_df_numeric, fill_value=0)

            # Calculate the average of the numeric columns
            df_numeric = df_numeric / len(files)
            print(len(files))

            # Combine the 'id' column back with the averaged numeric columns
            final_df = pd.concat([id_column, df_numeric], axis=1)

            # Save the result to a new CSV file
            print("the ensemble result is saved to ", os.path.join(result_dir,"ensemble_result.csv"))
            final_df.to_csv(os.path.join(result_dir,"ensemble_result.csv"), index=False)
        else:
            assert args.model_path is not None, "please provide the model path you want to test"
            model = TransformerClassifier().to(device)
            model.load_state_dict(torch.load(best_model_path))
            test(model, test_loader, test_ids, test_name = "best_model_test", result_dir = result_dir)  

            



    
        
    
 
    