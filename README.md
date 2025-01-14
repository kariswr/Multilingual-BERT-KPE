# **Multilingual-BERT-KPE** 

This project is a modification of **BERT for Keyphrase Extraction** (go to https://github.com/thunlp/BERT-KPE) so it runs in squad dataset with Multilingual BERT embedding model, and have an API endpoint to extract answer from a paragraph.\
This project is a part of Knowledge Self-Evaluation System with Automatic Factoid Question Generator.\
To know more about overall system : https://github.com/kariswr/Knowledge-Self-Evaluation-System.git

## How to Use

The example of preprocess training, and testing for squad dataset using Mutltilingual BERT Embedding model can be seen [here](https://github.com/kariswr/Multilingual-BERT-KPE/blob/master/notebook/Train%20Multilingual%20BERT-KPE%20All%20Epoch.ipynb).\
To get the squad dataset, unzip `BERT-Joint_squad_dataset.zip` in squad dataset folder.

### API

To run the API, use python api/api_endpoint.py in the repository folder.

The API will run at http://localhost:5001. \
To Extract Answer send a post request to  http://127.0.0.1:5001/extract-answer with body as follows:

```
{
    "paragraph" : "Dummy paragraph"
}
```

