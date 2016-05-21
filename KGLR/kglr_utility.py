'''
This file contains all the utility functions
'''
import re,sys,os,nltk,itertools
import MySQLdb as mdb
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.tag.stanford import StanfordPOSTagger
from kglr_settings import *
import kglr_settings

def get_lemmatizer():
    lemmatizer=WordNetLemmatizer()
    return lemmatizer

#convert POS treebank tag to wordnet tag
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return ''

#return data from a given table for a word
def get_thesaurus_db(word,table):
    if word+":"+table+":"+kglr_settings.search_str in kglr_settings.check_db_100.keys():
        return kglr_settings.check_db_100[word+":"+table+":"+kglr_settings.search_str]
    try:
        #time.sleep(0.1)
        con = mdb.connect('localhost', 'root', kglr_settings.password_db, kglr_settings.thesaurus_db)
        cur = con.cursor()
        word=word.replace("_"," ")
        word=word.replace("'","\\'")
        if kglr_settings.flag_pos_match and kglr_settings.search_str:
            sql="SELECT * FROM "+table+" WHERE Word='"+word+"' AND (SENSE REGEXP '^"+kglr_settings.search_str+"' OR SENSE REGEXP ', "+kglr_settings.search_str+":')"
        else:
            sql="SELECT * FROM "+table+" WHERE Word='"+word+"'"
        cur.execute(sql)
        result=cur.fetchall()
        if not result:
            if con:
                con.close()
            return []
        syn=[]
        for row in result :
            syn=syn+row[2].split(";")
        if con:
            con.close()
            
        if not(word+":"+table+":"+kglr_settings.search_str in kglr_settings.check_db_100.keys()):
            if len(kglr_settings.check_db_100)>=100:
                kglr_settings.check_db_100.popitem()
            kglr_settings.check_db_100[word+":"+table+":"+kglr_settings.search_str]=syn
        return syn
    
    except mdb.Error, e:
        print "Error %d: %s" % (e.args[0], e.args[1])
        sys.exit(1)

#returns number of hits for a word in a given table
def get_thesaurus_db_count(word,table):
    try:
        con = mdb.connect('localhost', 'root', kglr_settings.password_db, kglr_settings.thesaurus_db)
        cur = con.cursor()
        word=word.replace("_"," ")
        word=word.replace("'","\\'")
        if kglr_settings.flag_pos_match and kglr_settings.search_str:
            sql="SELECT * FROM "+table+" WHERE Word='"+word+"' AND (SENSE REGEXP '^"+kglr_settings.search_str+"' OR SENSE REGEXP ', "+search_str+":')"
        else:
            sql="SELECT * FROM "+table+" WHERE Word='"+word+"'"

        cur.execute(sql)
        result=cur.fetchall()
        if not result:
            if con:
                con.close()
            return 0
        else:
            return len(result)
        if con:
            con.close()
    
    except mdb.Error, e:
        print "Error %d: %s" % (e.args[0], e.args[1])
        sys.exit(1)

#returns number of synonyms (hits) for a word in 'synonyms' & 'syn_of_syn' table
def get_count(word):
    syn=get_thesaurus_db_count(word, "synonyms")
    if kglr_settings.flag_syn_of_syn:
        syn_of_syn= get_thesaurus_db_count(word, "syn_of_syn")
        syn=syn+syn_of_syn
    return syn

#returns synonyms(word) + syn_of_syn(word) - antonyms(word)
def get_synonyms(word_L):
    syn_all=[]
    for word in word_L:
        syn=get_thesaurus_db(word, "synonyms")
        if kglr_settings.flag_syn_of_syn:
            syn_of_syn= get_thesaurus_db(word, "syn_of_syn")
            syn=syn+syn_of_syn
        if kglr_settings.flag_antonyms:
            ant=get_thesaurus_db(word, "antonyms")
            #syn=list(set(syn)-set(ant))
            for ant_ele in ant:
                syn=filter(lambda a: a!=ant_ele,syn)
        syn_all=syn_all+syn
    return syn_all

#remove common preceding and trailing words in t1 & t2
def remove_common_affix(t1_L,t2_L,t1_P,t2_P):
    len_t1=len(t1_L)
    len_t2=len(t2_L)
    break_flag=0
    if len_t1<len_t2:
        min=len_t1
    else:
        min=len_t2
    for i in range(0,min):
        #print t2_L,t1_L,i
        if not t1_L[i]==t2_L[i]:
            break_flag=1
            break
    if break_flag:
        t1_P=t1_P[i:]
        t2_P=t2_P[i:]
        t1_L=t1_L[i:]
        t2_L=t2_L[i:]
    else:
        t1_P=t1_P[i+1:]
        t2_P=t2_P[i+1:]
        t1_L=t1_L[i+1:]
        t2_L=t2_L[i+1:]
        
    len_t1=len(t1_L)
    len_t2=len(t2_L)
    while(len_t1>0 and len_t2>0):
        if t1_L[len_t1-1]==t2_L[len_t2-1]:
            t1_P=t1_P[:-1]
            t2_P=t2_P[:-1]
            t1_L=t1_L[:-1]
            t2_L=t2_L[:-1]
        else:
            break
        len_t1=len_t1-1
        len_t2=len_t2-1
    return t1_L,t2_L,t1_P,t2_P

#Return a list of all possible derivations of a given word, while taking into account its pos
def get_derivation(in_word,in_word_P):
    derivation_L=[]
    if kglr_settings.flag_derivation:
        if kglr_settings.flag_pos_match and len(in_word.split())==1:
            if (re.search("VB",in_word_P[0].split("', '")[1].strip("('").strip("')"))):
                L_der=wn.synsets(in_word,pos="v")
            elif (re.search("NN",in_word_P[0].split("', '")[1].strip("('").strip("')"))):
                L_der=wn.synsets(in_word,pos="n")
            else:
                L_der=wn.synsets(in_word)
        else:
                L_der=wn.synsets(in_word)
        for ele in L_der:
            L_der_lem=ele.lemmas()
            for ele_1 in L_der_lem:
                derivation_L.append(ele_1.name().replace("_"," "))
                for ele_2 in ele_1.derivationally_related_forms():
                    derivation_L.append(ele_2.name().replace("_"," "))
        derivation_L.append(in_word.replace("_"," "))
        derivation_L=list(set(derivation_L))
    else:
        derivation_L.append(in_word)
    return(derivation_L)

#drops last word from t1&t2 # this function is called when the last words are IN and are synonymous
def drop_preposition(t1_L_rule,t2_L_rule,t1_P_rule,t2_P_rule):
    t1_L_rule=t1_L_rule[:-1]
    t2_L_rule=t2_L_rule[:-1]
    t1_P_rule=t1_P_rule[:-1]
    t2_P_rule=t2_P_rule[:-1] 
    return t1_L_rule,t2_L_rule,t1_P_rule,t2_P_rule

###Redundant preposition data###
def get_null_preposition_data():
    f_prep_null=open(kglr_settings.path+kglr_settings.redundant_prep_file,"r")
    prep_null={}
    for line_prep_null in f_prep_null:
        L_line_prep_null=line_prep_null.strip("\n").split("\t")
        if int(L_line_prep_null[2])>0:#modify
            prep_null_key=L_line_prep_null[0].split("_")[0].strip("'").strip('"')
            if prep_null_key in prep_null:
                temp_prep_null=prep_null[prep_null_key]
                temp_prep_null_data_L=L_line_prep_null[1].strip('"').strip("[").strip("]").strip("'").strip().split(",")
                temp_prep_null_data_L=[temp_prep_null_data_L_ele.strip().strip('"').strip("'") for temp_prep_null_data_L_ele in temp_prep_null_data_L]
                temp_prep_null_data_L+=temp_prep_null
                prep_null[prep_null_key]=list(set(temp_prep_null_data_L))
            else:
                temp_prep_null_data_L=L_line_prep_null[1].strip('"').strip("[").strip("]").strip("'").strip().split(",")
                temp_prep_null_data_L=[temp_prep_null_data_L_ele.strip().strip('"').strip("'") for temp_prep_null_data_L_ele in temp_prep_null_data_L]
                prep_null[prep_null_key]=list(set(temp_prep_null_data_L))
    f_prep_null.close()
    return prep_null

###Get Preposition Synonyms data###
def get_syn_preposition_data():
    f_prep=open(kglr_settings.path+kglr_settings.prep_syn_file)
    freq_prep={}
    for line_prep in f_prep:
        L_line_prep=line_prep.strip("\n").split(";")
        if int(L_line_prep[1])>20:
            freq_prep[L_line_prep[0]]=L_line_prep[1]
    f_prep.close()
    return freq_prep

###POS tag data###
def pos_tag_rules(lines): 
    stanford_pos_dict={}
    all_tok=[]
    for line in lines:
        line=line.strip("\n").replace("'","")
        L_line=line.split(";")
        t1=L_line[1].strip("@R@").replace("dont ","do not ").replace(" dont"," do not").replace("didnt ","do not ").replace(" didnt"," do not")
        t2=L_line[2].split("//")[0].strip("\n").strip("@R@").replace("dont ","do not ").replace(" dont"," do not").replace("didnt ","do not ").replace(" didnt"," do not")
        t1_L=t1.split()
        t2_L=t2.split()
        all_tok.append(t1_L)
        all_tok.append(t2_L)
    all_tok_pos=kglr_settings.st.tag_sents(all_tok)
    for all_tok_i in range(0,len(all_tok)):
        all_tok_str=" ".join(all_tok[all_tok_i])
        stanford_pos_dict[all_tok_str]=all_tok_pos[all_tok_i]
    return stanford_pos_dict
   
###read file###
def read_file(file_name):
    f=open(file_name)
    lines=f.readlines()
    f.close()
    return lines

###convert input rule string into a processable format###
def prepare_node(line,stanford_pos_dict):
    line=line.strip("\n").replace("'","");L_line=line.split(";")
    tag=L_line[0].strip('"')
    if tag=="1":
        kglr_settings.all_p=kglr_settings.all_p+1
    if tag=="2":
        kglr_settings.all_p_2=kglr_settings.all_p_2+1
        
    line_w=re.sub('";','",',line)
    
    #if <X;rel1;Y> --> <Y;rel2;X>: flag_rev_gate = 0 
    rev_R=L_line[1].endswith("@R@")
    rev_noR=L_line[2].split("//")[0].strip("\n").endswith("@R@")
    if kglr_settings.flag_rev:
        flag_rev_gate=(rev_R and rev_noR) or (not(rev_R) and not(rev_noR))
    else:
        flag_rev_gate=1
    
    #normalization
    t1=L_line[1].strip("@R@").replace("dont ","do not ").replace(" dont"," do not").replace("didnt ","do not ").replace(" didnt"," do not")
    t2=L_line[2].split("//")[0].strip("\n").strip("@R@").replace("dont ","do not ").replace(" dont"," do not").replace("didnt ","do not ").replace(" didnt"," do not")
    t1_L=t1.split()
    t2_L=t2.split()
                
    if kglr_settings.flag_stanford_pos:
        t1_P=stanford_pos_dict[" ".join(t1_L)]
        t2_P=stanford_pos_dict[" ".join(t2_L)]
        t1_P=[(str(ele[0]),str(ele[1])) for ele in t1_P]
        t2_P=[(str(ele[0]),str(ele[1])) for ele in t2_P]
    else:
        t1_P=nltk.pos_tag(t1_L)
        t2_P=nltk.pos_tag(t2_L)
        
    return line,L_line,line_w,flag_rev_gate,tag,t1,t1_L,t1_P,t2,t2_L,t2_P

#
def clean_data(t1_L,t1_P,t2_L,t2_P,line_w,line): 
    #1 word -- verb
    if len(t1_L)==1:
        t1_P[0]=t1_P[0][0]+":"+t1_P[0][1],"VB"
    if len(t2_L)==1:
        t2_P[0]=t2_P[0][0]+":"+t2_P[0][1],"VB"
    #len= 3 word - 1st word "be" and last word is a preposition then middle word is VB (given it has a verb form)
    if len(t1_P)>2:    
        if t1_L[0]=="be" and len(t1_L)==3 and t1_P[2][1]=="IN":
            word_pos_check_L=wn.synsets(t1_P[1][1])
            word_pos_check_L=[word_pos_check_L_ele.pos() for word_pos_check_L_ele in word_pos_check_L]
            if "v" in word_pos_check_L:
                t1_P[1]=t1_P[1][0]+":"+t1_P[1][1],"VB"
    if len(t2_P)>2:    
        if t2_L[0]=="be" and len(t2_L)==3 and t2_P[2][1]=="IN":
            word_pos_check_L=wn.synsets(t2_P[1][1])
            word_pos_check_L=[word_pos_check_L_ele.pos() for word_pos_check_L_ele in word_pos_check_L]
            if "v" in word_pos_check_L:
                t2_P[1]=t2_P[1][0]+":"+t2_P[1][1],"VB"
    
    ###mm - if 1st word is not be then it should be VB
    if not t1_L[0]=="be" and not re.search("^VB",t1_P[0][1]):
        t1_P[0]=t1_P[0][0]+":"+t1_P[0][1],"VB"
           
    if not t2_L[0]=="be" and not re.search("^VB",t2_P[0][1]):
        t2_P[0]=t2_P[0][0]+":"+t2_P[0][1],"VB"

    #trailing word cannot be noun
    ###mm - if 1st word is not 'be' then it should be VB
    if len(t1_L)>=2 and re.search("^NN",t1_P[len(t1_P)-1][1]):
        t1_P[len(t1_P)-1]=t1_P[len(t1_P)-1][0]+":"+t1_P[len(t1_P)-1][1],"VB"
           
    if len(t2_L)>=2 and re.search("^NN",t2_P[len(t2_P)-1][1]):
        t2_P[len(t2_P)-1]=t2_P[len(t2_P)-1][0]+":"+t2_P[len(t2_P)-1][1],"VB"

    t1_P=[str(ele) for ele in t1_P]
    t2_P=[str(ele) for ele in t2_P]
    
    t1_P_w=[str(ele).replace("', '", "'; '") for ele in t1_P]
    t2_P_w=[str(ele).replace("', '", "'; '")  for ele in t2_P]
    
    kglr_settings.f_w.write(line_w)
    kglr_settings.f_w.write(" , "+";".join(t1_P_w)+";".join(t2_P_w)+" , ")
    data_w=line+" , "+";".join(t1_P)+";".join(t2_P)+"\n"
    
    return t1_L,t1_P,t2_L,t2_P,data_w