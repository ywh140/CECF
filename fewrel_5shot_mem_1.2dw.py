import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import json
import gc
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm
from sklearn.cluster import KMeans
from encode import lstm_encoder, BERTSentenceEncoder
from dataprocess import data_sampler
from model import proto_softmax_layer
from dataprocess import get_data_loader
from transformers import BertTokenizer,BertModel
from util import set_seed,process_data,getnegfrombatch,select_similar_data_new,select_before_data_lstm_mem, npy2txt, compute_feature_by_data, get_match_id
import faiss
pad_token_label_id = nn.CrossEntropyLoss().ignore_index

def eval_model(config, basemodel, test_set, mem_relations):
    print("One eval")
    print("test data num is:\t",len(test_set))
    basemodel.eval()

    test_dataloader = get_data_loader(config, test_set, shuffle=False, batch_size=30)
    allnum= 0.0
    correctnum = 0.0
    for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext, lengths,
               typelabels) in enumerate(test_dataloader):
               
        logits, rep = basemodel(sentences, lengths)
        #print(logits.shape)  torch.Size([30,81])

        distances = basemodel.get_mem_feature(rep)
        short_logits = distances

        for index, logit in enumerate(logits):
            score = short_logits[index]  # logits[index] + short_logits[index] + long_logits[index]
            #print(score.shape)  (81,)
            allnum += 1.0

            golden_score = score[labels[index]]
            #print("<index>") 
            #print(index)   0~29
            #print(labels[index]) eg:tensor(17)
            #print(golden_score)  eg:0.9277003
            max_neg_score = -2147483647.0
            for i in neg_labels[index]:  # range(num_class):
                if (i != labels[index]) and (score[i] > max_neg_score):
                    max_neg_score = score[i]  #找到候选关系中最大的分数
            if golden_score > max_neg_score:  #如果候选关系中的分数都小于标签的分数，则正确
                correctnum += 1

    acc = correctnum / allnum
    print(acc)
    basemodel.train()
    return acc

def get_memory(config, model, proto_set):
    memset = []
    resset = []
    rangeset= [0]
    for i in proto_set:
        memset += i
        rangeset.append(rangeset[-1] + len(i))
    data_loader = get_data_loader(config, memset, False, False)
    features = []
    for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext, lengths,
               typelabels) in enumerate(data_loader):
        feature = model.get_feature(sentences, lengths)
        features.append(feature)
    features = np.concatenate(features)

    protos = []
    for i in range(len(proto_set)):
        protos.append(torch.tensor(features[rangeset[i]:rangeset[i+1],:].mean(0, keepdims = True)))
    protos = torch.cat(protos, 0)
    return protos

def select_data(mem_set, proto_memory, config, model, divide_train_set, num_sel_data, current_relations, selecttype):
    ####select data according to selecttype
    #selecttype is 0: cluster for every rel
    #selecttype is 1: use ave embedding
    rela_num = len(current_relations)
    for i in range(0, rela_num):
        thisrel = current_relations[i]
        if thisrel in mem_set.keys():
            #print("have set mem before")
            mem_set[thisrel] = {'0': [], '1': {'h': [], 't': []}}
            proto_memory[thisrel].pop()
        else:
            mem_set[thisrel] = {'0': [], '1': {'h': [], 't': []}}
        thisdataset = divide_train_set[thisrel]
        data_loader = get_data_loader(config, thisdataset, False, False)
        features = []
        for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext,  lengths,
                   typelabels) in enumerate(data_loader):
            feature = model.get_feature(sentences, lengths)
            features.append(feature)
        features = np.concatenate(features)
        #print(features.shape)
        num_clusters = min(num_sel_data, len(thisdataset))
        if selecttype == 0:
            kmeans = KMeans(n_clusters=num_clusters, random_state=0)
            distances = kmeans.fit_transform(features)
            for i in range(num_clusters):
                sel_index = np.argmin(distances[:, i])
                instance = thisdataset[sel_index]
                ###change tylelabel
                instance[11] = 3
                ###add to mem data
                mem_set[thisrel]['0'].append(instance)  ####positive sample
                cluster_center = kmeans.cluster_centers_[i]
                #print(cluster_center.shape)
                proto_memory[thisrel].append(instance)
        elif selecttype == 1:
            #print("use average embedding")
            samplenum = features.shape[0]
            veclength = features.shape[1]
            sumvec = np.zeros(veclength)
            for j in range(samplenum):
                sumvec += features[j]
            sumvec /= samplenum

            ###find nearest sample
            mindist = 100000000
            minindex = -100
            for j in range(samplenum):
                dist = np.sqrt(np.sum(np.square(features[j] - sumvec)))
                if dist < mindist:
                    minindex = j
                    mindist = dist
            #print(minindex)
            instance = thisdataset[j]
            ###change tylelabel
            instance[11] = 3
            mem_set[thisrel]['0'].append(instance)
            proto_memory[thisrel].append(instance)
        else:
            print("error select type")
    #####to get negative sample  mem_set[thisrel]['1']
    if rela_num > 1:
        ####we need to sample negative samples
        allnegres = {}
        for i in range(rela_num):
            thisnegres = {'h':[],'t':[]}
            currel = current_relations[i]
            thisrelposnum = len(mem_set[currel]['0'])
            #assert thisrelposnum == num_sel_data
            #allnum = list(range(thisrelposnum))
            for j in range(thisrelposnum):
                thisnegres['h'].append(mem_set[currel]['0'][j][3])
                thisnegres['t'].append(mem_set[currel]['0'][j][5])
            allnegres[currel] = thisnegres
        ####get neg sample
        for i in range(rela_num):
            togetnegindex = (i + 1) % rela_num
            togetnegrelname = current_relations[togetnegindex]
            mem_set[current_relations[i]]['1']['h'].extend(allnegres[togetnegrelname]['h'])
            mem_set[current_relations[i]]['1']['t'].extend(allnegres[togetnegrelname]['t'])
    return mem_set

tempthre = 0.2

def train_model_with_hard_neg(config, model, mem_set, traindata, epochs, current_proto, ifnegtive=0):
    #print("<traindata>")
    #print(len(traindata))
    #print(len(train_set))
    mem_data = []
    #print("mem_set")
    #print(mem_set)
    if len(mem_set) != 0:
        for key in mem_set.keys():
            mem_data.extend(mem_set[key]['0'])
    #print(len(mem_data))
    train_set = traindata + mem_data
    #print(len(train_set))
    data_loader = get_data_loader(config, train_set, batch_size=config['batch_size_per_step'])
    model.train()
    criterion = nn.CrossEntropyLoss()
    lossfn = nn.MultiMarginLoss(margin=0.2)
    optimizer = optim.Adam(model.parameters(), config['learning_rate'])
    for epoch_i in range(epochs):
        model.set_memorized_prototypes(current_proto)
        losses1 = []
        losses2 = []
        losses3 = []
        losses4 = []
        losses5 = []

        lossesfactor1 = 0.0
        lossesfactor2 = 1.0
        lossesfactor3 = 1.0
        lossesfactor4 = 1.0
        lossesfactor5 = 0.1
        loss_print = 0
        count_loss = 0
        for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext, lengths,
                   typelabels) in enumerate(data_loader):
            model.zero_grad()
            #print(len(sentences))
            labels = labels.to(config['device'])
            typelabels = typelabels.to(config['device'])  ####0:rel  1:pos(new train data)  2:neg  3:mem
            numofmem = 0
            numofnewtrain = 0
            allnum = 0
            memindex = []
            for index,onetype in enumerate(typelabels):
                if onetype == 1:
                    numofnewtrain += 1
                if onetype == 3:
                    numofmem += 1
                    memindex.append(index)
                allnum += 1
            #print(numofmem)
            #print(numofnewtrain)
            getnegfromnum = 1
            allneg = []
            alllen = []
            if numofmem > 0:
                ###select neg data for mem
                for oneindex in memindex:
                    negres,lenres = getnegfrombatch(oneindex,firstent,firstentindex,secondent,secondentindex,sentences,lengths,getnegfromnum,allnum,labels,neg_labels)
                    for aa in negres:
                        allneg.append(torch.tensor(aa))
                    for aa in lenres:
                        alllen.append(torch.tensor(aa))
            sentences.extend(allneg)
            lengths.extend(alllen)
            logits, rep = model(sentences, lengths)
            #print(logits.shape)
            #print(rep.shape)
            logits_proto = model.mem_forward(rep)
            #print(logits_proto.shape)
            logitspos = logits[0:allnum,]
            #print(logitspos.shape)
            logits_proto_pos = logits_proto[0:allnum,]
            #print(logits_proto_pos.shape)
            if numofmem > 0:
                logits_proto_neg = logits_proto[allnum:,]

            logits = logitspos
            logits_proto = logits_proto_pos
            loss1 = criterion(logits, labels)
            loss2 = criterion(logits_proto, labels)
            loss4 = lossfn(logits_proto, labels)
            loss3 = torch.tensor(0.0).to(config['device'])
            for index, logit in enumerate(logits):
                score = logits_proto[index]
                preindex = labels[index]
                maxscore = score[preindex]
                size = score.shape[0]
                secondmax = -100000
                for j in range(size):
                    if j != preindex and score[j] > secondmax:
                        secondmax = score[j]
                if secondmax - maxscore + tempthre > 0.0:
                    loss3 += (secondmax - maxscore + tempthre).to(config['device'])
            loss3 /= logits.shape[0]

            start = 0
            loss5 = torch.tensor(0.0).to(config['device'])
            allusenum = 0
            for index in memindex:
                onepos = logits_proto[index]
                posindex = labels[index]
                #poslabelscore = torch.exp(onepos[posindex])
                poslabelscore = onepos[posindex]
                negnum = getnegfromnum * 2
                negscore = torch.tensor(0.0).to(config['device'])
                for ii in range(start, start + negnum):
                    oneneg = logits_proto_neg[ii]
                    #negscore += torch.exp(oneneg[posindex])
                    negscore = oneneg[posindex]
                    if negscore - poslabelscore + 0.01 > 0.0 and negscore < poslabelscore:
                        loss5 += (negscore - poslabelscore + 0.01)
                        allusenum += 1
                #loss5 += (-torch.log(poslabelscore/(poslabelscore+negscore)))
                start += negnum
            #print(len(memindex))
            if len(memindex) == 0:
                loss = loss1 * lossesfactor1 + loss2 * lossesfactor2 + loss3 * lossesfactor3 + loss4 * lossesfactor4
            else:
                #loss5 /= len(memindex)
                loss5 = loss5 / allusenum
                #loss = loss1 * lossesfactor1 + loss2 * lossesfactor2 + loss3 * lossesfactor3 + loss4 * lossesfactor4    ###no loss5
                loss = loss1 * lossesfactor1 + loss2 * lossesfactor2 + loss3 * lossesfactor3 + loss4 * lossesfactor4 + loss5 * lossesfactor5    ###with loss5
            
            loss_print = loss_print + loss
            count_loss = count_loss + 1

            loss.backward()
            losses1.append(loss1.item())
            losses2.append(loss2.item())
            losses3.append(loss3.item())
            losses4.append(loss4.item())
            losses5.append(loss5.item())
            #print("step:\t", step, "\tloss1:\t", loss1.item(), "\tloss2:\t", loss2.item(), "\tloss3:\t", loss3.item(),
            #      "\tloss4:\t", loss4.item(), "\tloss5:\t", loss5.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()

        # print("hard neg model loss:%f" % (loss_print/count_loss))
        return model

def train_simple_model(currentnumber, beforenumber, iterate, pos_matrix, O_pos_matrix, match_id, match_O_id, steps, refer_model, config, model, mem_set, train_set, epochs, current_proto, ifusemem=False):
    #loss列表
    distill_list, ce_list = [], []
    if ifusemem:
        mem_data = []
        if len(mem_set)!=0:
            for key in mem_set.keys():
                mem_data.extend(mem_set[key]['0'])
        train_set.extend(mem_data)
    model.train()
    data_loader = get_data_loader(config, train_set, batch_size = config['batch_size_per_step']) #batch_size=5
    criterion = nn.CrossEntropyLoss()
    lossfn = nn.MultiMarginLoss(margin=0.2)
    optimizer = optim.Adam(model.parameters(), config['learning_rate'])
    for epoch_i in range(epochs):
        model.set_memorized_prototypes(current_proto)
        losses1 = []
        losses2 = []
        losses3 = []
        losses4 = []

        lossesfactor1 = 0.0
        lossesfactor2 = 1.0
        lossesfactor3 = 1.0
        lossesfactor4 = 1.0
        loss_print = 0
        count_loss = 0

        sample_id, O_sample_id, sentence_id = 0, 0, 0
        top_k = 3

        for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext,
                   lengths, typelabels) in enumerate(tqdm(data_loader)):
            model.zero_grad()
            logits, rep = model(sentences, lengths)
            logits_proto = model.mem_forward(rep)

            labels = labels.to(config['device'])
            loss1 = criterion(logits, labels)
            loss2 = criterion(logits_proto, labels)
            loss4 = lossfn(logits_proto, labels)
            loss3 = torch.tensor(0.0).to(config['device'])

            ####添加因果效应损失
            if  steps>0:
                #batch_size = 8 batch_sent_ids = [0, 1, 2, 3, 4, 5, 6, 7]
                batch_sent_ids = list(range(sentence_id,sentence_id+len(sentences)))
                #num_samples_batch = batch_size
                typelabels_list = typelabels.numpy().tolist()
                num_samples_batch = config['batch_size_per_step']-typelabels_list.count(-100)
                #计算该batch的句子对应match的句子id
                match_id_batch = match_id[sample_id*top_k:(sample_id+num_samples_batch)*top_k]
                sample_id += num_samples_batch
                #print("<match_id_batch>")
                #print(match_id_batch)
                #计算features_match和logits_match
                if match_id_batch!=None and len(match_id_batch)==0:
                    features_match = None
                    logits_match = None
                elif match_id_batch!=None and len(match_id_batch)>0:
                    select_sentence_batch = []
                    select_sentence_batch_lengths = []
                    select_sentence_idx = []
                    for i in range(len(match_id_batch)):
                        select_sentence_idx.append(pos_matrix[match_id_batch[i]])
                    for idx in select_sentence_idx:
                        select_sentence_batch.append(torch.tensor(train_set[idx][2])) #把选中的句子添加进去
                        select_sentence_batch_lengths.append(torch.tensor(train_set[idx][10]))
                    #print("<select_sentence_batch>")
                    #print(select_sentence_batch)
                    with torch.no_grad():
                        model.eval()
                        tmp_features_match_lst = []
                        tmp_logists_match_lst = []
                        for i in range(0, len(select_sentence_batch), 30):  #每30个分一组
                            if i+30 > len(select_sentence_batch):
                                _select_batch = select_sentence_batch[i:]
                                _select_batch_lengths = select_sentence_batch_lengths[i:]
                            else:
                                _select_batch = select_sentence_batch[i:i+30]
                                _select_batch_lengths = select_sentence_batch_lengths[i:i+30]
                            tmp_logists_match, tmp_features_match = model(_select_batch, _select_batch_lengths)
                            tmp_features_match_lst.append(tmp_features_match) #torch.Size(batch_num, 句子长度, hidden_dim)
                            tmp_logists_match_lst.append(tmp_logists_match)
                        tmp_features_match = torch.cat(tmp_features_match_lst, dim=0)  #torch.Size(batch_num, 句子长度, hidden_dim)
                        tmp_logists_match = torch.cat(tmp_logists_match_lst, dim=0)
                        features_match = torch.FloatTensor(len(match_id_batch),tmp_features_match.shape[-1])
                        logits_match = torch.cuda.FloatTensor(tmp_logists_match)
                        #print("logits_match")
                        #print(logits_match)
                        #print(logits_match.shape)

                O_batch_sent_ids = list(range(sentence_id,sentence_id+len(sentences)))
                num_O_samples_batch = typelabels_list.count(-100)
                O_match_id_batch = match_O_id[O_sample_id*top_k:(O_sample_id+num_O_samples_batch)*top_k]
                O_sample_id += num_O_samples_batch

                if O_match_id_batch!=None and len(O_match_id_batch)==0:
                    O_features_match = None
                    O_logits_match = None
                elif O_match_id_batch!=None and len(O_match_id_batch)>0:
                    O_select_sentence_batch = []
                    O_select_sentence_batch_lengths = []
                    O_select_sentence_idx = []
                    for i in range(len(O_match_id_batch)):
                        O_select_sentence_idx.append(O_pos_matrix[O_match_id_batch[i]])
                    #print("O_match_id_batch")
                    #print(O_match_id_batch)
                    #print("O_select_sentence_idx")
                    #print(O_select_sentence_idx)
                    for idx in O_select_sentence_idx:
                        O_select_sentence_batch.append(torch.tensor(train_set[idx][2])) #把选中的句子添加进去
                        O_select_sentence_batch_lengths.append(torch.tensor(train_set[idx][10]))
                    #print("<O_select_sentence_batch>")
                    #print(O_select_sentence_batch)
                    with torch.no_grad():
                        model.eval()
                        O_tmp_features_match_lst = []
                        O_tmp_logists_match_lst = []
                        for i in range(0, len(O_select_sentence_batch), 30):  #每30个分一组
                            if i+30 > len(O_select_sentence_batch):
                                O_select_batch = O_select_sentence_batch[i:]
                                O_select_batch_lengths = O_select_sentence_batch_lengths[i:]
                            else:
                                O_select_batch = O_select_sentence_batch[i:i+30]
                                O_select_batch_lengths = O_select_sentence_batch_lengths[i:i+30]
                            O_tmp_logists_match, O_tmp_features_match = model(O_select_batch, O_select_batch_lengths)
                            O_tmp_features_match_lst.append(O_tmp_features_match) #torch.Size(batch_num, hidden_dim)
                            O_tmp_logists_match_lst.append(O_tmp_logists_match)
                        O_tmp_features_match = torch.cat(O_tmp_features_match_lst, dim=0)  #torch.Size(batch_num, hidden_dim)
                        O_tmp_logists_match = torch.cat(O_tmp_logists_match_lst, dim=0)
                        O_features_match = torch.FloatTensor(len(O_match_id_batch),O_tmp_features_match.shape[-1])
                        O_logits_match = torch.cuda.FloatTensor(O_tmp_logists_match)
                        #print("O_logits_match")
                        #print(O_logits_match)
                        #print(O_logits_match.shape)

                #ce_loss, distill_loss = batch_loss_distill(logits_match, labels, logits, refer_model, model, sentences)
                ce_loss, distill_loss = batch_loss_distill(logits_match, O_logits_match, labels, logits, refer_model, model, sentences, lengths, typelabels)

            model.train() 

            mask = torch.zeros_like(typelabels)
            mask = torch.logical_or(mask, typelabels!=-100)
              
            ####add triple loss
            for index, logit in enumerate(logits):
                score = logits_proto[index]
                preindex = labels[index]
                maxscore = score[preindex]
                size = score.shape[0]
                secondmax = -100000
                for j in range(size):
                    if j != preindex and score[j] > secondmax:
                        secondmax = score[j]
                if secondmax - maxscore + tempthre > 0.0:
                    loss3 += (secondmax - maxscore + tempthre).to(config['device'])

            loss3 /= logits.shape[0]
            # loss = loss1 * lossesfactor1 + loss2 * lossesfactor2 + loss3 * lossesfactor3 + loss4 * lossesfactor4
            if steps > 0:
                distill_weight = config['distill_weight']*np.power((currentnumber)/(beforenumber),0.5)
                #loss = loss1 * lossesfactor1 + loss3 * lossesfactor3 + loss4 * lossesfactor4 + ce_loss + distill_loss
                loss = loss1 * lossesfactor1 + loss3 * lossesfactor3 + loss4 * lossesfactor4 + ce_loss + distill_weight*distill_loss
            else:
                loss = loss1 * lossesfactor1 + loss2 * lossesfactor2 + loss3 * lossesfactor3 + loss4 * lossesfactor4
            
            #if iterate == 0 or iterate == 3 or iterate == 5:
            #    if steps > 0:
            #       loss = loss1 * lossesfactor1 + loss2 * lossesfactor2 + loss3 * lossesfactor3 + loss4 * lossesfactor4 + ce_loss + distill_loss
            #    else:
            #        loss = loss1 * lossesfactor1 + loss2 * lossesfactor2 + loss3 * lossesfactor3 + loss4 * lossesfactor4
            #if iterate == 1 or iterate ==2:
            #    if steps > 0:
            #        loss = loss1 * lossesfactor1 + loss2 * lossesfactor2 + loss3 * lossesfactor3 + loss4 * lossesfactor4 + ce_loss + 2.0*distill_loss
            #    else:
            #        loss = loss1 * lossesfactor1 + loss2 * lossesfactor2 + loss3 * lossesfactor3 + loss4 * lossesfactor4
            #if iterate == 4:
            #    if steps > 0:
            #        loss = loss1 * lossesfactor1 + loss3 * lossesfactor3 + loss4 * lossesfactor4 + ce_loss + 2.0*distill_loss
            #    else:
            #        loss = loss1 * lossesfactor1 + loss2 * lossesfactor2 + loss3 * lossesfactor3 + loss4 * lossesfactor4


            loss_print = loss_print + loss
            count_loss = count_loss + 1
            #print(" loss:%f" % loss)

            loss.backward()
            losses1.append(loss1.item())
            losses2.append(loss2.item())
            losses3.append(loss3.item())
            losses4.append(loss4.item())
            if steps > 0:
                ce_list.append(ce_loss.item())
                distill_list.append(distill_loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
        
        print("simple model loss:%f" % (loss_print/count_loss))
        #print (np.array(losses).mean())
    return model

def batch_loss_distill(logits_match, O_logits_match, labels, logits, refer_model, model, sentences, lengths, typelabels):

    refer_dims = refer_model.output_dim
    all_dims = model.output_dim

    # Check input
    assert logits!=None, "logits is none!"
    assert refer_model!=None, "refer_model is none!"
    assert sentences!=None, "inputs is none!"
    assert len(sentences)==len(labels), "inputs and labels are not matched!"

    # (1) CE loss
    ce_mask = torch.zeros_like(typelabels)
    ce_mask = torch.logical_or(ce_mask, typelabels!=-100)
    if logits_match!=None:
        ce_loss = compute_DCE(config, labels, logits, logits_match, ce_mask)
    elif torch.sum(ce_mask.float())==0:
        ce_loss = torch.tensor(0., requires_grad=True).cuda()
    else:
        ce_loss = compute_CE(labels, logits, ce_mask)
    
    # （2）Distill loss
    distill_mask = torch.zeros_like(typelabels)
    distill_mask = torch.logical_or(distill_mask, typelabels==-100)
    with torch.no_grad():
        refer_model.eval()
        refer_logits, refer_features = refer_model(sentences, lengths)
        assert refer_logits.shape == logits.shape, \
                "the first 2 dims of refer_logits and logits are not equal!!!"
    if O_logits_match!= None:
        distill_loss = compute_ODCE(config, refer_logits, logits, O_logits_match, distill_mask, refer_dims)
    elif torch.sum(distill_mask.float())==0:
        distill_loss = torch.tensor(0., requires_grad=True).cuda()
    else:
        distill_loss = compute_KLDiv(config, refer_logits, logits, distill_mask)

    #print("ce_loss")
    #print(ce_loss)
    #print("distill_loss")
    #print(distill_loss)
    return ce_loss, distill_loss


def compute_CE(labels, logits, ce_mask):
    '''
        Cross-Entropy Loss
    '''
    all_dims = logits.shape[-1]
    ce_loss = nn.CrossEntropyLoss()(logits[ce_mask].view(-1, all_dims),
                            labels[ce_mask].flatten().long())
    return ce_loss

def compute_KLDiv(config, refer_logits, logits, distill_mask):
    '''
        KLDivLoss
    '''
    refer_dims = refer_logits.shape[-1]

    # 1.log(distribution)
    old_class_score = F.log_softmax(
                        logits[distill_mask]/config['temperature'],
                        dim=-1)[:,:refer_dims].view(-1, refer_dims)
    # 2.ref_distribution
    ref_old_class_score = F.softmax(
                        refer_logits[distill_mask]/config['ref_temperature'], 
                        dim=-1).view(-1, refer_dims)

    distill_loss = nn.KLDivLoss(reduction='batchmean')(old_class_score, ref_old_class_score)

    return distill_loss


def compute_DCE(config, labels, logits, logits_match, ce_mask):
    '''
        DCE for labeled samples
    '''
    # joint ce_loss
    #对每行进行softmax归一化（dim=-1）
    logits_prob = F.softmax(logits, dim=-1) #self.logits=torch.Size([batch_size, max_length, all_dims]) 
    #self.logits.view(-1,self.logits.shape[-1])=torch.Size([batch_size*max_length, all_dims]) 
    logits_prob_match = F.softmax(logits_match, dim=-1) #torch.Size([实体数*top_k(2*3), all_dims])
    #print("logits_prob_match")
    #print(logits_prob_match)
    logits_prob_match = torch.mean(logits_prob_match.reshape(-1, config['top_k'], logits_prob_match.size(-1)), dim=1) #取平均数 torch.Size([实体数（2）, all_dims])
    logits_prob_joint = (logits_prob[ce_mask] + logits_prob_match)/2 #把原本的logits和平均后的match_logists合并平均

    ce_loss = F.nll_loss(torch.log(logits_prob_joint+1e-10), labels[ce_mask])

    return ce_loss

def compute_ODCE(config, refer_logits, logits, O_logits_match, distill_mask, refer_dims):
    '''
        DCE for labeled samples
    '''
    refer_dims = refer_logits.shape[-1]
    O_logits_prob_match = F.softmax(
                            O_logits_match/config['temperature'], 
                            dim=-1)
    #O_logits_prob_match = F.softmax(
    #                        O_logits_match[:,:refer_dims]/config['temperature'], 
    #                        dim=-1)

    # O_logits_prob_matc[15,81];O_logits_prob_match.view(-1, config['top_k'], O_logits_prob_match.shape[-1]) [5,3,81]
    O_logits_prob_match = torch.mean(O_logits_prob_match.view(-1, config['top_k'], O_logits_prob_match.shape[-1]), dim=1)
     
    old_class_score_all = F.softmax(
                            logits/config['temperature'],
                            dim=-1)
    #old_class_score_all = F.softmax(
    #                        logits/config['temperature'],
    #                        dim=-1)[:,:refer_dims]
    #print("<old_class_score_all>")
    #print(old_class_score_all.shape)
    joint_old_class_score_all = old_class_score_all.clone()

    # curriculum learning
    if config['is_curriculum_learning']:
        # select the samples with highest prob
        assert epoch!=-1, "Epoch should be given for curriculum learning!!!"
        prob_threshold = get_cl_prob_threshold(config, epoch) 

        curriculum_mask = torch.max(old_class_score_all[defined_O_mask],dim=-1)[0]>=prob_threshold  
        # old_class_score_all[defined_O_mask]:defined_O_sample单词的logits
        # torch.max[0]每一维最大的logits [1]每一维最大的logits的位置
        # curriculum_mask:最大的logits >= prob_threshold为True
        defined_O_curricumlum_mask = defined_O_mask.clone()
        for i in range(defined_O_curricumlum_mask.shape[0]):
            for j in range(defined_O_curricumlum_mask.shape[1]):
                if defined_O_curricumlum_mask[i][j] and torch.max(old_class_score_all[i][j])<prob_threshold:
                    defined_O_curricumlum_mask[i][j]=False
        # defined_O_curricumlum_mask:defined_O_sample中logits<prob_threshold：False, logits>=rob_threshold：True

        # Compute KL divergence of distributions
        # 1.log(joint_distribution)
        joint_old_class_score_all[defined_O_curricumlum_mask] = (old_class_score_all[defined_O_curricumlum_mask]+O_logits_prob_match[curriculum_mask])/2
        joint_old_class_score = torch.log(joint_old_class_score_all[distill_mask]+1e-10).view(-1, refer_dims)
        # 2.ref_distribution
        # Sharpen the effect of the define O samples
        cl_temperature = self.get_cl_temperature(self.epoch)
        refer_logits[defined_O_curricumlum_mask] /= cl_temperature
        undefined_O_mask = torch.logical_and(distill_mask,torch.logical_not(defined_O_curricumlum_mask))
        refer_logits[undefined_O_mask] /= self.params.ref_temperature
        ref_old_class_score = F.softmax(
                                refer_logits[distill_mask], 
                                dim=-1).view(-1, refer_dims)
        # KL divergence
        distill_loss = nn.KLDivLoss(reduction='batchmean')(joint_old_class_score, ref_old_class_score)
    else:
        # TODO: KLDiv or CE ?
        # Compute KL divergence of distributions
        # 1.log(joint_distribution)
        # print(O_logits_prob_match)
        # print(old_class_score_all[defined_O_mask])
        joint_old_class_score_all[distill_mask] = (old_class_score_all[distill_mask]+O_logits_prob_match)/2
        joint_old_class_score = torch.log(joint_old_class_score_all[distill_mask]+1e-10).view(-1, refer_dims)
        # 2.ref_distribution
        refer_logits[distill_mask] /= 1e-10 # Equals to applying CE to defined O samples, others is KLDivLoss
        ref_old_class_score = F.softmax(
                            refer_logits[distill_mask]/config['ref_temperature'], 
                            dim=-1).view(-1, refer_dims)
        # KL divergence
        distill_loss = nn.KLDivLoss(reduction='batchmean')(joint_old_class_score, ref_old_class_score)
        #print("<distill_loss>")
        #print(distill_loss)
        
    return distill_loss

def get_cl_prob_threshold(config, epoch):
    epoch_s, epoch_e = config['cl_epoch_start'], config['cl_epoch_end']    # cl_epoch_start=1 cl_epoch_end=10 epoch_s=1 epoch_e=10
    pro_s, pro_e = config['cl_prob_start'], config['cl_prob_end']    # cl_prob_start=0.9 cl_prob_end=0 pro_s=0.9 pro_e=0
    if epoch<epoch_s:
        return pro_s
    elif epoch>epoch_e:
        return pro_e
    else:
        return pro_s+(epoch-epoch_s)/(epoch_e-epoch_s)*(pro_e-pro_s)

def get_cl_temperature(config, epoch):
    epoch_s, epoch_e = config['cl_epoch_start'], config['cl_epoch_end']
    tmp_s, tmp_e = config['cl_tmp_start'], config['cl_tmp_end']          # cl_tmp_start: 0.001, cl_tmp_end: 0.001
    if epoch<epoch_s:
        return tmp_s
    elif epoch>epoch_e:
        return tmp_e
    else:
        return np.power(10,np.log10(tmp_s)+(epoch-epoch_s)/(epoch_e-epoch_s)*(np.log10(tmp_e)-np.log10(tmp_s)))

if __name__ == '__main__':

    select_thredsold_param = 0.65
    select_num = 1
    f = open("config/config_fewrel_5and10.json", "r")
    #f = open("config/config_bert.json", "r")
    config = json.loads(f.read())
    f.close()
    config['device'] = torch.device('cuda' if torch.cuda.is_available() and config['use_gpu'] else 'cpu')
    config['n_gpu'] = torch.cuda.device_count()
    config['batch_size_per_step'] = int(config['batch_size'] / config["gradient_accumulation_steps"])
    config['neg_sampling'] = False
    config['is_curriculum_learning'] = False
    config['temperature'] = 1
    config['top_k'] = 3
    config['ref_temperature'] = 1
    config['distill_weight'] = 1.2
    config['average'] = True

    root_path = '.'
    word2id = json.load(open(os.path.join(root_path, 'glove/word2id.txt')))
    word2vec = np.load(os.path.join(root_path, 'glove/word2vec.npy'))

    #'''
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    donum = 1

    distantpath = "data/distantdata/"
    file1 = distantpath + "distant.json"
    file2 = distantpath + "exclude_fewrel_distant.json"
    list_data,entpair2scope = process_data(file1,file2)

    topk = 16
    max_sen_length_for_select = 64
    max_sen_lstm_tokenize = 128
    select_thredsold = select_thredsold_param

    print("********* load from ckpt ***********")
    ckptpath = "simmodelckpt"
    #ckptpath = "/data/qin/FewShotContinualRE/mycode/newcode/ckpt_of_step_40000"
    print(ckptpath)
    ckpt = torch.load(ckptpath)
    SimModel = BertModel.from_pretrained('bert-base-uncased',state_dict=ckpt["bert-base"], local_files_only=True).to(config["device"])

    allunlabledata = np.load("allunlabeldata.npy").astype('float32')

    d = 768 * 2
    index = faiss.IndexFlatIP(d)
    print(index.is_trained)
    index.add(allunlabledata)  # add vectors to the index
    print("<index.ntotal>")
    print(index.ntotal)

    for m in range(donum):
        print(m)
        config["rel_cluster_label"] = "data/fewrel/CFRLdata_10_100_10_5/rel_cluster_label_" + str(m) + ".npy"
        config['training_file'] = "data/fewrel/CFRLdata_10_100_10_5/train_" + str(m) + ".txt"
        config['valid_file'] = "data/fewrel/CFRLdata_10_100_10_5/valid_" + str(m) + ".txt"
        config['test_file'] = "data/fewrel/CFRLdata_10_100_10_5/test_" + str(m) + ".txt"

        encoderforbase = lstm_encoder(token2id=word2id, word2vec=word2vec, word_size=len(word2vec[0]), max_length=128, pos_size=None,
                                    hidden_size=config['hidden_size'], dropout=0, bidirectional=True, num_layers=1, config=config)
        #encoderforbase = BERTSentenceEncoder(config=config)
        sampler = data_sampler(config, encoderforbase.tokenizer)

        # 转化数据
        unlable_data = npy2txt(index, encoderforbase.tokenizer, list_data)

        modelforbase = proto_softmax_layer(encoderforbase, num_class=len(sampler.id2rel), id2rel=sampler.id2rel, drop=0, config=config)
        modelforbase = modelforbase.to(config["device"])

        word2vec_back = word2vec.copy()

        sequence_results = []
        result_whole_test = []
        for i in range(6):

            num_class = len(sampler.id2rel)
            print(config['random_seed'] + 10 * i)
            set_seed(config, config['random_seed'] + 10 * i)
            sampler.set_seed(config['random_seed'] + 10 * i)

            mem_set = {} ####  mem_set = {rel_id:{'0':[positive samples],'1':[negative samples]}} 换5个head 换5个tail
            mem_relations = []   ###not include relation of current task
            iterate = i

            past_relations = []

            savetest_all_data = None
            saveseen_relations = []

            proto_memory = []

            for k in range(len(sampler.id2rel)):
                proto_memory.append([sampler.id2rel_pattern[k]])
            oneseqres = []
            ##################################
            whichdataselecct = 1
            ifnorm = True
            ##################################
            for steps, (training_data, valid_data, test_data, test_all_data, seen_relations, current_relations) in enumerate(sampler):
                
                #定义refer_model
                if steps == 0:
                    refer_model = None
                else:
                    refer_model = deepcopy(modelforbase)
                    refer_model.eval()

                best_score = -1

                #print(steps)\
                print("<training_data>")
                print(len(training_data))
                #for aa in range(20):
                #    print(training_data[aa])
                savetest_all_data = test_all_data
                saveseen_relations = seen_relations
                modelforbase.output_dim = len(seen_relations)

                ##修
                beforenumber = 0
                if steps > 0:
                    before_relations = seen_relations[:-10]
                    print("<before_relations>")
                    print(before_relations)
                    beforenumber = len(before_relations)
                ##

                currentnumber = len(current_relations)
                #print("<current number>")
                #print(currentnumber)
                print("<current relations>")
                print(current_relations)
                divide_train_set = {}
                for relation in current_relations:
                    divide_train_set[relation] = []  ##int
                for data in training_data:
                    divide_train_set[data[0]].append(data)
                print(len(divide_train_set))

                ####select most similar sentence for new task, not for base task

                ####step==0是base model
                if steps == 0:
                    ##train base model
                    print("train base model,not select most similar")
                    logits_match = 0
                    match_id = None
                    match_O_id = None
                    pos_matrix = []
                    O_pos_matrix = []
                    training_data_add = training_data

                else:
                    print("train new model,select most similar")

                    selectdata = select_similar_data_new(training_data, tokenizer, entpair2scope, topk,
                                                            max_sen_length_for_select,list_data, config, SimModel,
                                                            select_thredsold,max_sen_lstm_tokenize,encoderforbase.tokenizer,index,ifnorm,select_num)
                    training_data.extend(selectdata)
                    data_num = len(training_data)
                    print("<add_training_data>")
                    print(data_num)
                    #print(training_data)
                    data_loader = get_data_loader(config, training_data, shuffle=False, batch_size=30)

                    #利用上一个模型识别无标签的文本并标记
                    labeled_data = select_before_data_lstm_mem(config, before_relations, refer_model, unlable_data, mem_set)
                    #print(labeled_data)
                    add_data_num = len(labeled_data)
                    print("<add data num>")
                    print(add_data_num)

                    #data_loader
                    labeled_data_loader = get_data_loader(config, labeled_data, shuffle=False, batch_size=30)

                    pos_matrix = []
                    for a in range(data_num):
                        pos_matrix.extend([a])
                    O_pos_matrix = []
                    for a in range(add_data_num):
                        O_pos_matrix.extend([a+data_num])

                    #计算训练数据的feature
                    refer_flatten_feat_train = compute_feature_by_data(config, data_loader, refer_model, current_relations, is_normalize=True)
                    #print("<refer_flatten_feat_train>")
                    #print(refer_flatten_feat_train.shape)
                    refer_flatten_feat_O_train = compute_feature_by_data(config, labeled_data_loader, refer_model, before_relations, is_normalize=True)
                    #print("<refer_flatten_feat_O_train>")
                    #print(refer_flatten_feat_O_train)
                    #print(refer_flatten_feat_O_train.shape)

                    match_id = None
                    match_O_id = None
                    #计算每个sample的邻居
                    match_id = get_match_id(config, refer_flatten_feat_train, config['top_k'],)
                    #print("匹配的match_id", match_id.shape)
                    #print(match_id)
                    match_O_id = get_match_id(config, refer_flatten_feat_O_train, config['top_k'],)
                    #print("匹配的match_O_id", match_O_id.shape)
                    #print("<match_O_id>")
                    #print(match_O_id)
                    #print(match_O_id.shape)

                    #将识别到的文本加入训练数据中
                    training_data_add = deepcopy(training_data)
                    training_data_add.extend(labeled_data)

                if config['average']:
                    current_proto = get_memory(config, modelforbase, proto_memory)
                    modelforbase = train_simple_model(currentnumber, beforenumber, iterate, pos_matrix, O_pos_matrix, match_id, match_O_id, steps, refer_model, config, modelforbase, mem_set, training_data_add, 1, current_proto, False)
            
                    select_data(mem_set, proto_memory, config, modelforbase, divide_train_set,
                                config['rel_memory_size'], current_relations, 0)  ##config['rel_memory_size'] == 1

                    #######select_data_whole(mem_set, proto_memory, config, modelforbase, divide_train_set,config['rel_memory_size'] * len(current_relations), current_relations)

                    for j in range(2):
                        current_proto = get_memory(config, modelforbase, proto_memory)
                        modelforbase = train_model_with_hard_neg(config, modelforbase, mem_set, training_data, 1,
                                                             current_proto, ifnegtive=0)


                    current_proto = get_memory(config, modelforbase, proto_memory)
                    modelforbase.set_memorized_prototypes(current_proto)
                    mem_relations.extend(current_relations)

                    currentalltest = []
                    for mm in range(len(test_data)):
                        currentalltest.extend(test_data[mm])
                        #eval_model(config, modelforbase, test_data[mm], mem_relations)

                    thisstepres = eval_model(config, modelforbase, currentalltest, mem_relations)
                    print("step:\t",steps,"\taccuracy:\t",thisstepres)
                else:
                    for k in range(4):

                        model_ckpt_name = "iteration_%s_steps_%d_k_%d.pth"%(
                                        i, 
                                        steps,
                                        k)
                        model_ckpt_path = os.path.join(
                            '/root/ywh/', 
                            model_ckpt_name
                        )
                        print("model_ckpt_path")
                        print(model_ckpt_path)

                        if k==0:
                            current_proto = get_memory(config, modelforbase, proto_memory)
                        modelforbase = train_simple_model(pos_matrix, O_pos_matrix, match_id, match_O_id, steps, refer_model, config, modelforbase, mem_set, training_data_add, 1, current_proto, False)
            
                        if k==0:
                            select_data(mem_set, proto_memory, config, modelforbase, divide_train_set,
                                        config['rel_memory_size'], current_relations, 0)  ##config['rel_memory_size'] == 1

                        #######select_data_whole(mem_set, proto_memory, config, modelforbase, divide_train_set,config['rel_memory_size'] * len(current_relations), current_relations)

                        for j in range(2):
                            current_proto = get_memory(config, modelforbase, proto_memory)
                            modelforbase = train_model_with_hard_neg(config, modelforbase, mem_set, training_data, 1,
                                                                 current_proto, ifnegtive=0)

                        if k==0:
                            current_proto = get_memory(config, modelforbase, proto_memory)
                            modelforbase.set_memorized_prototypes(current_proto)
                            mem_relations.extend(current_relations)

                            currentalltest = []
                            for mm in range(len(test_data)):
                                currentalltest.extend(test_data[mm])
                            #eval_model(config, modelforbase, test_data[mm], mem_relations)

                        thisstepres = eval_model(config, modelforbase, currentalltest, mem_relations)
                        print("step:\t",steps,"\taccuracy:\t",thisstepres)
                        
                        if thisstepres > best_score:
                            print("Find better model!!")
                            best_score = thisstepres
                            torch.save(modelforbase, model_ckpt_path)
                            best_model_ckpt_path = model_ckpt_path
                            print("Best model has been saved to %s" % model_ckpt_path)


                if config['average']:
                    oneseqres.append(thisstepres)
                else:
                    oneseqres.append(best_score)
                    modelforbase =torch.load(best_model_ckpt_path)
                    modelforbase.eval()
                    thisstepres = eval_model(config, modelforbase, currentalltest, mem_relations)
                    print("Load model from %s" % best_model_ckpt_path)
                    print("step:\t",steps,"\taccuracy:\t",thisstepres)

            sequence_results.append(np.array(oneseqres))

            #def eval_both_model(config, newmodel, basemodel, test_set, mem_relations, baserelation, newrelation, proto_embed):
            allres = eval_model(config, modelforbase, savetest_all_data, saveseen_relations)
            result_whole_test.append(allres)

            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            print("after one epoch allres:\t",allres)
            print(result_whole_test)

            # initialize the models
            modelforbase = modelforbase.to('cpu')
            del modelforbase
            gc.collect()
            if config['device'] == 'cuda':
                torch.cuda.empty_cache()
            encoderforbase = lstm_encoder(token2id=word2id, word2vec=word2vec_back.copy(), word_size=len(word2vec[0]),max_length=128, pos_size=None,
                                          hidden_size=config['hidden_size'], dropout=0, bidirectional=True, num_layers=1, config=config)
            modelforbase = proto_softmax_layer(encoderforbase, num_class=len(sampler.id2rel), id2rel=sampler.id2rel,
                                               drop=0, config=config)
            modelforbase.to(config["device"])
            # output the final avg result
        print("Final result!")
        print(result_whole_test)
        for one in sequence_results:
            for item in one:
                sys.stdout.write('%.4f, ' % item)
            print('')
        avg_result_all_test = np.average(sequence_results, 0)
        for one in avg_result_all_test:
            sys.stdout.write('%.4f, ' % one)
        print('')
        print("Finish training............................")
    #'''

