<h2>PT</h2>
STAT3612 project, PT: A Plain Transformer is Hospital Readmission Predictor. 

This is a very rough implementation of the paper, I try my best to make the code readable and easy to understand. 



<h3>Installation</h3>

```bash 
conda env create -f environment.yml
conda activate pt
```

<h3>Usage</h3>
place the data into the same folder so that the main structure of this folder is as follows:

```
-PT 
    #results
    -pre-run-result
    -results
    #code
    Data.py 
    main.py 
    model.py
    #data 
    notes.csv 
    ehr_preprocessed_seq_by_day_cat_embedding.pkl
```



Extract tfidf features from the data taking few minutes to run, ensure that notes.csv is in the same folder as the code 


```bash
python note_feature_extraction_t.py
```

```bash
python get_note_label.py
```





Test best model without training 


```bash
python main.py --task test --ensemble 0  --model_path pre-run-result/best_model_test.pth
```

Test with ensemble model without training
Note that this result will be stored in pre-run-result/ensemble, it's different from other results

```bash 
python PT_main.py --task test --ensemble 1 --result_dir "pre-run-result/ensemble"
```

Train and test model without ensemble 

```bash
python main.py
```

Train and test model with ensemble

```bash
python PT_main.py --ensemble 1
```

The validation set is a bit difficult to get a good result, the best validation AUC is above 0.87 without out ensemble model
The ensemble model's validation AUC will be around 0.86 - 0.91



Expected results: 
without ensemble: 0.89 AUC on public, 0.88 AUC on private
with ensemble: 0.90 AUC on public, 0.905 AUC on private



For grader:
Any questions, please contact at: 
fanz1@connect.hku.hk

Or raise an issue on github https://github.com/IILKA/PT






