## Setup

### 1. Download the code


### 2. Install modified transformers

```
cd transformers; pip install .; cd ..
```

### 3. Download pre-trained files
we use the pre-trained files provided by our baseline.
please refer to the code of the paper "Continual Few-shot Relation Learning via Embedding Space Regularization and Data Augmentation" for details.


### 4. Run script

```
python fewrel_5shot_mem_1.2dw.py
```

### 5. Attention: The problems you may encounter.
#### transformer
'''
pip uninstall transformers
pip install transformers==4.5.0
'''

#### wordninja
'''
pip install wordninja
'''

#### faiss
'''
conda install -c pytorch faiss-gpu cudatoolkit=11.0
'''

#### faiss wrong (python3.9 -> 3.8)
'''
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
unzip libstdc.so_.6.0.26.zip
sudo mv libstdc++.so.6.0.26  /usr/lib/x86_64-linux-gnu/
cd /usr/lib/x86_64-linux-gnu/
sudo rm libstdc++.so.6
ln libstdc++.so.6.0.26 libstdc++.so.6
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
'''

#### version
/root/miniconda3/envs/myconda/lib/python3.8/site-packages/transformers/dependency_versions_table.py
    "tokenizers": "tokenizers>=0.10.1",
