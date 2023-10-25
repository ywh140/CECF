import numpy as np
import faiss
from util import set_seed,process_data,getnegfrombatch,select_similar_data_new,select_before_data_lstm


distantpath = "data/distantdata/"
file1 = distantpath + "distant.json"
file2 = distantpath + "exclude_fewrel_distant.json"
list_data,entpair2scope = process_data(file1,file2)

allunlabeldata = np.load("allunlabeldata.npy").astype('float32')
d = 768 * 2
index = faiss.IndexFlatIP(d)
print(index.is_trained)
index.add(allunlabeldata)  # add vectors to the index
print(index.ntotal)
for i in range(index.ntotal):
    onenum = i
    onesen = " ".join(list_data[onenum]["tokens"])
    ###handle onesen
    onesen.replace("\n\n\n", " ")
    onesen.replace("\n\n", " ")
    onesen.replace("\n", " ")
    print("<onesen>")
    print(onesen)
