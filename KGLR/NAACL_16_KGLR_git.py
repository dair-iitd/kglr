'''
Created on Dec 24, 2014-start 7 jan

@author: Prachi

COde to apply rules on i/p entailments from jonathan

1) scrape missing words using create_thesaurus_syn_of_syn.py (ensure you add the new function 
you added in this code there as well)
2) copy the newly scraped data from thesaurus2.txt to thesaurus.txt and same for antonyms and syn_of_syn
3) run populate_db.py to store the new data in mysql db as well
4) run this code to apply rules.
'''
#utility#core [opertor]#read-write
import re,sys,os,nltk,itertools
import MySQLdb as mdb
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn

###Set paths###
java_path = "C:\\Program Files\\Java\\jdk1.8.0_60\\bin\\java.exe"#/usr/lib/jvm/java-8-oracle/bin/java
postagger_model_path='E:\\nltk_data\\stanford\\stanford-postagger.jar'
distsimtagger_model_path='E:\\nltk_data\\stanford\\model\\english-bidirectional-distsim.tagger'
#('/home/prachi/nltk_data/stanford/model/english-bidirectional-distsim.tagger','/home/prachi/nltk_data/stanford/stanford-postagger.jar', java_options='-mx20G')
#project folder
path="E:\\EclipseIndigo\\workspace\\Inference\\"#path="/home/prachi/Documents/project/code/ver_18_5/NLP_TE/"
#input file
rule_in_file="in_files\\ablation.txt"#rule_in_file="in_files/ablation.txt"
#output file
rule_out_file=""
#Data files#
redundant_prep_file="data\\ch9out2-r-0000all_norm_IN_null.txt"#redundant_prep_file="data/ch9out2-r-0000all_norm_IN_null.txt"
prep_syn_file="data\\prep_pair_all_freq.txt"#prep_syn_file="data/prep_pair_all_freq.txt"
java_options_mem='-mx2G'
#thesaurus db
thesaurus_db = 'thesaurus'
password_db=''
###Set paths###
os.environ['JAVAHOME'] = java_path
from nltk.tag.stanford import StanfordPOSTagger#POSTagger
st=StanfordPOSTagger(distsimtagger_model_path,postagger_model_path, java_options=java_options_mem)
#('/home/prachi/nltk_data/stanford/model/english-bidirectional-distsim.tagger','/home/prachi/nltk_data/stanford/stanford-postagger.jar', java_options='-mx20G')
lemmatizer=WordNetLemmatizer()

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

check_thesaurus_100={}
check_db_100={}
count_t2int1_prev=0
flag_affix_call=0

aux_verb=["be"]
light_verb_old=["take", "have", "give", "do", "make"]
verbs_verb=["has","have","be","is","were","are","was","had","being","began","am","following","having","do","does",
            "did","started","been","became","left","help","helped","get","keep","think","got","gets","include",
            "suggest","used","see","consider","means","try","start","included","lets","say","continued",
            "go","includes","becomes","begins","keeps","begin","starts","said"]#,"stop"
verb_verb_norm=["begin","start","continue","say"]
all_verb_verb=verb_verb_norm+verbs_verb

flag_stanford_pos=1

flag_whole=1
flag_not=1
flag_affix=1
flag_be_trail=1
#flag_dt=1
flag_deep_dt=1
flag_antonyms=1
flag_syn_of_syn=1

flag_noun_verb_wn=1
flag_rules=1
flag_gerund_infi_to=1
rules_dt=1
rule_be=1
rule_have=1
rule_equal_preposition=1
rule_equal_preposition_data_freq=1
rule_preposition_null=1#X;learn about;Y --> X;learn;Y

rule_active_passive_be=1
rule_have_JJ=1
rule_be_prep=1
rule_lightverb_dt=1
rule_superlative=1
rule_JJ_NN_mod=1
rule_verbverb=1
#flag to check t1 in t2'syn and t2 in t1's syn
flag_bidir_syn=1
# output in t1 in t2 file
flag_pos_match=1
#flag_multiple_word=1#to match each word in t1 with each word in t2
#TO FIX :: Why writing twice and at arbit places
#To FIX:: make sure the rule doesnt apply for whole word checking!!
#To FIX:: count the number of "0" in o/p file and the ones printed!
flag_level2_hypernyms=1

flag_rev=1#to ensure: X t1 Y -> X t2 Y
flag_wordnet_hypernym_hyponym=1

flag_derivation=1

tp_2=0
tp=0
fp=0
all_p=0
all_p_2 = 0
flag_positive=1

L_gerund_infi_to=["attempt","begin","bother","cease","continue","deserve","neglect","omit","permit","start","fear","intend","recommend","advice","allow","permit","encourage","forbid","choose"]
L_gerund_infi_to_t1=["like","love","prefer"]

def nounify_guided(verb_word,noun_word):
    set_of_related_nouns = set()
    if wn.morphy(verb_word, wn.VERB):
        for lemma in wn.lemmas(wn.morphy(verb_word, wn.VERB), pos="v"):
            for related_form in lemma.derivationally_related_forms():
                for synset in wn.synsets(related_form.name(), pos=wn.NOUN):
                    if wn.synset('person.n.01') in synset.closure(lambda s:s.hypernyms()):
                        if synset.lemmas()[0].name()==noun_word:
                            return 1
                    #set_of_related_nouns.add(synset)

    return 0

def check_preposition_equivalence(t1,t2,equal_preposition,level,t1_P_rule,t2_P_rule):
    for prep_pair in equal_preposition:
        x=len(t1)-len(prep_pair[0].split())
        y=len(t1)
        if x>0:
            w1_L=t1[x:y]
        else:
            w1_L=t1[y-1]
        x=(len(t2)-len(prep_pair[1].split()))
        y=len(t2)
        if x>0:
            w2_L=t2[x:y]
        else:
            w2_L=t2[y-1]
        w1=" ".join(w1_L)
        w2=" ".join(w2_L)
        if w1==prep_pair[0] and w2==prep_pair[1]:
            t1=t1[:-len(prep_pair[0].split())]
            t2=t2[:-len(prep_pair[1].split())]
            t1_P_rule=t1_P_rule[:-len(prep_pair[0].split())]
            t2_P_rule=t2_P_rule[:-len(prep_pair[1].split())]
            return True,t1,t2,t1_P_rule,t2_P_rule,prep_pair
        elif not(level>1):
            check_1,t1,t2,t1_P_rule,t2_P_rule,prep_pair=check_preposition_equivalence(t1, t2, [(prep_pair[1],prep_pair[0])],2,t1_P_rule,t2_P_rule)
            if check_1:
                return True,t1,t2,t1_P_rule,t2_P_rule,prep_pair
    return False,t1,t2,t1_P_rule,t2_P_rule,prep_pair     

def get_thesaurus_db(word,table):
    if word+":"+table+":"+search_str in check_db_100.keys():
        return check_db_100[word+":"+table+":"+search_str]
    try:
        #time.sleep(0.1)
        con = mdb.connect('localhost', 'root', password_db, thesaurus_db)
        cur = con.cursor()
        word=word.replace("_"," ")
#        if re.search("'s chips",word):
#            print "catch"
        word=word.replace("'","\\'")
        if flag_pos_match and search_str:
            sql="SELECT * FROM "+table+" WHERE Word='"+word+"' AND (SENSE REGEXP '^"+search_str+"' OR SENSE REGEXP ', "+search_str+":')"
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
            #print row[0]," : ",row[1]," : ",row[2]," : ",row[3],"---\n",row
        #syn=syn.remove("")
        if con:
            con.close()
            
        if not(word+":"+table+":"+search_str in check_db_100.keys()):
            if len(check_db_100)>=100:
                check_db_100.popitem()
            check_db_100[word+":"+table+":"+search_str]=syn
        return syn
    
    except mdb.Error, e:
        print "Error %d: %s" % (e.args[0], e.args[1])
        sys.exit(1)
#    finally:
#        if con:
#            con.close()

def old_get_thesaurus_db(word,table):
    if word+":"+table in check_db_100.keys():
        return check_db_100[word+":"+table]
    try:
        #time.sleep(0.1)
        con = mdb.connect('localhost', 'root', password_db, thesaurus_db)
        cur = con.cursor()
        word=word.replace("_"," ")
#        if re.search("'s chips",word):
#            print "catch"
        word=word.replace("'","\\'")
        if flag_pos_match and search_str:
            sql="SELECT * FROM "+table+" WHERE Word='"+word+"' AND (SENSE REGEXP '^"+search_str+"' OR SENSE REGEXP ', "+search_str+":')"
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
            #print row[0]," : ",row[1]," : ",row[2]," : ",row[3],"---\n",row
        #syn=syn.remove("")
        if con:
            con.close()
            
        if not(word+":"+table in check_db_100.keys()):
            if len(check_db_100)>=100:
                check_db_100.popitem()
            check_db_100[word+":"+table]=syn
        return syn
    
    except mdb.Error, e:
        print "Error %d: %s" % (e.args[0], e.args[1])
        sys.exit(1)
#    finally:
#        if con:
#            con.close()

def get_thesaurus_db_count(word,table):
    try:
        con = mdb.connect('localhost', 'root', password_db, thesaurus_db)
        cur = con.cursor()
        word=word.replace("_"," ")
#        if re.search("'s chips",word):
#            print "catch"
        word=word.replace("'","\\'")
#        sql="SELECT * FROM "+table+" WHERE Word='"+word+"'"
        if flag_pos_match and search_str:
            #if re.search("verb",search_str) or re.search("adj",search_str):
            #    sql="SELECT * FROM "+table+" WHERE Word='"+word+"' AND (SENSE REGEXP '^"+"(adj|verb)"+"' OR SENSE REGEXP ', "+search_str+":')"
            #else:
            sql="SELECT * FROM "+table+" WHERE Word='"+word+"' AND (SENSE REGEXP '^"+search_str+"' OR SENSE REGEXP ', "+search_str+":')"
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
#    finally:
#        if con:
#            con.close()
def get_count(word):
    syn=get_thesaurus_db_count(word, "synonyms")
    if flag_syn_of_syn:
        syn_of_syn= get_thesaurus_db_count(word, "syn_of_syn")
        syn=syn+syn_of_syn
    return syn

def get_synonyms(word_L):
    syn_all=[]
    for word in word_L:
        syn=get_thesaurus_db(word, "synonyms")
        if flag_syn_of_syn:
            syn_of_syn= get_thesaurus_db(word, "syn_of_syn")
            syn=syn+syn_of_syn
        if flag_antonyms:
            ant=get_thesaurus_db(word, "antonyms")
            #syn=list(set(syn)-set(ant))
            for ant_ele in ant:
                syn=filter(lambda a: a!=ant_ele,syn)
        syn_all=syn_all+syn
    return syn_all

search_str=""

morphy_tag = {'NN':wn.NOUN,'JJ':wn.ADJ,'VB':wn.VERB,'RB':wn.ADV}

def check_relation(t1_P_clip,t1_L_clip,t2_L_clip,t2_P_clip,level,flag_positive,prev_dir,f_th,data_w,direction,flag_write):
    pos_t1=""
    if len(t1_P_clip)==1:
        for morphy_tag_key in morphy_tag:
            if re.search("^"+morphy_tag_key,t1_P_clip[0].split("', '")[1].strip("('").strip("')")):
                pos_t1=morphy_tag[morphy_tag_key]
                break
    
    pos_t2=""
    if len(t2_P_clip)==1:
        for morphy_tag_key in morphy_tag:
            if re.search("^"+morphy_tag_key,t2_P_clip[0].split("', '")[1].strip("('").strip("')")):
                pos_t2=morphy_tag[morphy_tag_key]
                break
    if pos_t1:
        t1_syn=wn.synsets(t1_L_clip,pos_t1)
    else:
        t1_syn=wn.synsets(t1_L_clip)
    if pos_t2:
        t2_syn=wn.synsets(t2_L_clip,pos_t2)
    else:
        t2_syn=wn.synsets(t2_L_clip)
    flag_hypernym=0
    flag_hyponym=0
    
    flag_common_hypernym=0
    
    t1_syn_lemma=[]
    for ele in t1_syn:
        t1_syn_lemma+=ele.lemmas()
    t2_syn_lemma=[]
    for ele in t2_syn:
        t2_syn_lemma+=ele.lemmas()
    t2_syn_lemma_str=[str(ele) for ele in t2_syn_lemma]
    t1_syn_lemma_str=[str(ele) for ele in t1_syn_lemma]
    for t1_syn_ele in t1_syn:
        t1_syn_ele_hyper=t1_syn_ele.hypernyms()
        t1_syn_ele_hyper_lemma=[]
        for ele in t1_syn_ele_hyper:
            t1_syn_ele_hyper_lemma+=ele.lemmas()
            if flag_level2_hypernyms:
                ele_level2_hyp=ele.hypernyms()
                for ele_level2_hyp_ele in ele_level2_hyp:
                    t1_syn_ele_hyper_lemma+=ele_level2_hyp_ele.lemmas()
        t1_syn_ele_hyper_lemma_str=[str(ele) for ele in t1_syn_ele_hyper_lemma]
        if set(t1_syn_ele_hyper_lemma_str).intersection(set(t2_syn_lemma_str)):
            flag_hypernym=1
            break
    return flag_hypernym,flag_hyponym,flag_common_hypernym

def check_thesaurus(t1_P_clip,t1_L_clip,t2_L_clip,t2_P_clip,level,flag_positive,prev_dir,f_th,data_w,direction,flag_write):
    global count_t2int1_prev,search_str
    flag_found=0
    
    #if re.search("be interrupted by@R@;be disturbed",line_w):
    #    print "dd"
    
    search_str=""
    if rule_verbverb and not(direction=="rev"):#not(t1_P_clip==["dummy"]):
        t1_P_clip,t1_L_clip,flag_found=check_verbverb_construction(t1_P_clip,t1_L_clip,t2_L_clip)
        if not flag_positive and t1_L_clip and t2_L_clip: 
            t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L_clip, t2_L_clip,t1_P_clip,t2_L_clip)
            if not(t1_affix) and not(t2_affix):
                flag_positive=1
                f_w_rule_t2_in_t1.write(data_w)
                        
    if type(t1_L_clip) is list:
        t1_L_clip=" ".join(t1_L_clip)
                
    temp_t1_L_clip=t1_L_clip.split()#fixed 21 may 16: (t1_L_clip.split())
    lemm_pos=""
    search_str_1=""
    search_str_2=""
                
    if len(temp_t1_L_clip)==1:
        if flag_pos_match and (re.search("\(",t1_P_clip[0])):
            if (re.search("VB",t1_P_clip[0].split("', '")[1].strip("('").strip("')"))):# or re.search("be "+t1_L_clip[0],line)):
                search_str="verb"
                lemm_pos=wn.VERB
            elif (re.search("NN",t1_P_clip[0].split("', '")[1].strip("('").strip("')"))):# or re.search("be "+t1_L_clip[0],line)):
                    search_str="noun"
                    lemm_pos=wn.NOUN
            elif (re.search("JJ",t1_P_clip[0].split("', '")[1].strip("('").strip("')"))):# or re.search("be "+t1_L_clip[0],line)):
                    search_str="adj"
        if lemm_pos:
            temp_t1_L_clip.append(lemmatizer.lemmatize(t1_L_clip, lemm_pos))
        else:
            temp_t1_L_clip.append(lemmatizer.lemmatize(t1_L_clip))
            
        temp_t1_L_clip.append(lemmatizer.lemmatize(t1_L_clip))
        temp_t1_L_clip=list(set(temp_t1_L_clip))
    else:
        temp_t1_L_clip=[t1_L_clip]
    
    if type(t2_L_clip) is list:
        t2_L_clip=" ".join(t2_L_clip)
    temp_t2_L_clip=(t2_L_clip.split())
    if len(temp_t2_L_clip)==1:
        if flag_pos_match:
            if (re.search("VB",t2_P_clip[0].split("', '")[1].strip("('").strip("')"))):# or re.search("be "+t2_L_clip[0],line)):
                lemm_pos=wn.VERB
                search_str_2="verb"
            elif (re.search("NN",t2_P_clip[0].split("', '")[1].strip("('").strip("')"))):# or re.search("be "+t1_L_clip[0],line)):
                lemm_pos=wn.NOUN
                search_str_2="noun"#"verb"
            elif (re.search("JJ",t2_P_clip[0].split("', '")[1].strip("('").strip("')"))):# or re.search("be "+t1_L_clip[0],line)):
                search_str_2="adj"
        if lemm_pos:
            temp_t2_L_clip.append(lemmatizer.lemmatize(t2_L_clip, lemm_pos))
        else:
            temp_t2_L_clip.append(lemmatizer.lemmatize(t2_L_clip))
        
        temp_t2_L_clip.append(lemmatizer.lemmatize(t2_L_clip, wn.VERB))
        temp_t2_L_clip.append(lemmatizer.lemmatize(t2_L_clip))
        temp_t2_L_clip=list(set(temp_t2_L_clip))
    else:
        temp_t2_L_clip=[t2_L_clip]
    count_t2int1=0
    search_str_1=search_str
    if (re.search("verb",search_str_1) and re.search("adj",search_str_2)) or (re.search("verb",search_str_2) and re.search("adj",search_str_1)):
        search_str="(adj|verb)"
        search_str_2=search_str_1=search_str
    tmp_list=get_synonyms(temp_t1_L_clip)
    search_str=""

#     m_del1=0
#     m_flag_positive=0
#     m_prev_dir=0
#     if not flag_positive and flag_multiple_word and len(t1_L_clip.split())>1 and (len(t1_L_clip.split())==len(t2_L_clip.split()))  and (len(t1_P_clip)==len(t2_P_clip)) and len(t1_P_clip)>1:
#         flag_write=0
#         #m_del1,m_flag_positive,m_prev_dir=del1,flag_positive,prev_dir
#         for m_len_i in range(0,len(t1_L_clip.split())):
#             #if m_len_i==len(t1_L_clip.split())-1:
#             #    flag_write=1
#             t1_L_clip_m=t1_L_clip.split()
#             t2_L_clip_m=t2_L_clip.split()
#             m_del1,m_flag_positive,m_prev_dir=check_thesaurus([t1_P_clip[m_len_i]], t1_L_clip_m[m_len_i], t2_L_clip_m[m_len_i], [t2_P_clip[m_len_i]], 1, flag_positive, prev_dir, f_th, data_w, direction,flag_write)
#             if m_flag_positive==0:
#                 break
#             if direction=="rev":
#                 direction_m="fwd"
#             else:
#                 direction_m="rev"
#             if m_len_i==len(t1_L_clip.split())-1:
#                 flag_write=1
#             
#             m_del1,m_flag_positive,m_prev_dir=check_thesaurus([t2_P_clip[m_len_i]], t2_L_clip_m[m_len_i], t1_L_clip_m[m_len_i], [t1_P_clip[m_len_i]], 2, m_flag_positive, m_prev_dir, f_th, data_w, direction_m,flag_write)
#             
#             if m_flag_positive==0:
#                 break
#         del1,flag_positive,prev_dir=m_del1,m_flag_positive,m_prev_dir
#         if m_flag_positive:
#             return(del1,flag_positive,prev_dir)


    if len(tmp_list)>0:#t1_L_clip)>0:
        if len(t2_L_clip.split())>0 and len(t1_L_clip.split())>0:                
            #tmp_list=thesaurus[t1_L_clip].split(";")
            if len(tmp_list)>1:
                flag_t2_L_clip_in_tmp_list=False
                for temp_t2_L_clip_ele in temp_t2_L_clip:
                    if temp_t2_L_clip_ele in tmp_list:
                        flag_t2_L_clip_in_tmp_list=True
                        count_t2int1=+tmp_list.count(temp_t2_L_clip_ele)
                        #break
                del1=str(flag_t2_L_clip_in_tmp_list)
            else:
                del1="False"
        else:
            del1="False"
        
        if level==1:
            count_t2int1_prev=count_t2int1                    
            if del1=="True":
                prev_dir=1
                if flag_positive==0 and not flag_bidir_syn:
                    flag_positive=1
                    #f_th.write(str(count_t2int1)+":"+data_w)
                    f_th.write(data_w)
                    f_th.flush()
                if flag_positive==0 and flag_found and flag_bidir_syn and rule_verbverb:
                    del1,flag_positive,prev_dir=check_thesaurus(t2_P_clip,t2_L_clip,t1_L_clip,t1_P_clip,level,flag_positive,prev_dir,f_th,data_w,"rev",flag_write)
#            if len(check_thesaurus_100)>=100:#not(str(t1_P_clip)+":"+str(t1_L_clip)+":"+str(t2_L_clip) in check_thesaurus_100):
#                check_thesaurus_100.popitem()
#            check_thesaurus_100[str(t1_P_clip)+":"+str(t1_L_clip)+":"+str(t2_L_clip)+":"+str(level)+":"+str(flag_positive)+":"+str(prev_dir)]=del1,flag_positive,prev_dir
            
            return(del1,flag_positive,prev_dir)       
        elif level ==2:
            if del1=="True":
                if flag_bidir_syn and prev_dir and not flag_positive:
                    #print line
                    search_str=search_str_1
                    #if re.search("be taken on",line_w):
                    #    print "12"
                    #print t1_L_clip,search_str
                    count_t1=get_count(t1_L_clip)#+0.00000001
                    search_str=search_str_2
                    count_t2=get_count(t2_L_clip)#+0.00000001                  
                    search_str=""
                    
                    if flag_affix_call==0:
                        #f_th.write(str((float(count_t2int1_prev)+float(count_t2int1))/(count_t1*count_t2))+":"+str(count_t2int1_prev)+":"+str(count_t2int1)+":"+data_w)
                        f_th.write(data_w)
                        #f_th.write(str(count_t2int1_prev)+":"+str(count_t2int1)+":"+data_w)
                        flag_positive=1
                    else:
                        #print count_t2int1_prev,count_t2int1,count_t1,count_t2
                        #print t1_P_clip,t2_P_clip
                        if count_t1 and count_t2:
                            count_data_val=(float(count_t2int1_prev)+float(count_t2int1))/float(count_t1*count_t2)#count_data_val=(float(count_t2int1_prev)+float(count_t2int1))/float(count_t1+count_t2)#/float(count_t1*count_t2)
                        else:
                            count_data_val=0
                        if count_data_val >=0.003:#count_data_val>=0.1:#count_data_val >=0.003:
                            #f_th.write(str((float(count_t2int1_prev)+float(count_t2int1))/(count_t1*count_t2))+":"+str(count_t2int1_prev)+":"+str(count_t2int1)+":"+data_w)
                            #f_th.write(str(count_data_val)+" :: "+data_w)
                            f_th.write(data_w)
                            #f_th.write(str(count_data_val)+" :: "+str(count_t2int1_prev)+" :: "+str(count_t2int1)+" :: "+str(count_t1)+" :: "+str(count_t2)+" :: "+data_w)
                            #f_th.write(str(count_t2int1_prev)+":"+str(count_t2int1)+":"+data_w)
                            flag_positive=1
                        else:
                            flag_positive=0
                    f_th.flush()
                if flag_positive==0 and flag_found and flag_bidir_syn and rule_verbverb:
                    prev_dir=1
                    del1,flag_positive,prev_dir=check_thesaurus(t2_P_clip,t2_L_clip,t1_L_clip,t1_P_clip,level,flag_positive,prev_dir,f_th,data_w,"rev",flag_write)
#            
            prev_dir=0
            count_t2int1_prev=0                                
            return(del1,flag_positive,prev_dir)         
    else:
        prev_dir=0
        del1="False"
        if len(check_thesaurus_100)>=100:
            check_thesaurus_100.popitem()
        check_thesaurus_100[str(t1_P_clip)+":"+str(t1_L_clip)+":"+str(t2_L_clip)+":"+str(level)+":"+str(flag_positive)+":"+str(prev_dir)]=del1,flag_positive,prev_dir
            
        return(del1,flag_positive,prev_dir)

def check_lightverb_construction(t1_P_lightverbDT,t1_L_lightverbDT):
    flag_found=0
    if len(t1_P_lightverbDT)>2 and len(t1_L_lightverbDT)>2:
        if (t1_L_lightverbDT[0] in light_verb_old) and (re.search("^(a|an)",t1_P_lightverbDT[1].split("', '")[0].strip("('").strip("')"))) and (re.search("^(NN|VB)",t1_P_lightverbDT[2].split("', '")[1].strip("('").strip("')"))):
            t1_P_lightverbDT=t1_P_lightverbDT[2:]
            t1_L_lightverbDT=t1_L_lightverbDT[2:]
            flag_found=1
    return t1_P_lightverbDT,t1_L_lightverbDT,flag_found

def check_verbverb_construction(t1_P_lightverbDT,t1_L_lightverbDT,t2_L_lightverbDT):
    #if re.search("move towards",line):
    #    print "dghj"
    flag_found=0
    if not type(t1_L_lightverbDT) is list:
        t1_L_lightverbDT=t1_L_lightverbDT.split()
    if not type(t2_L_lightverbDT) is list:
        t2_L_lightverbDT=t2_L_lightverbDT.split()
    if len(t1_P_lightverbDT)>1 and len(t1_L_lightverbDT)>1:
        if (t1_L_lightverbDT[0] in all_verb_verb) and ((re.search("^(NN|VB)",t1_P_lightverbDT[1].split("', '")[1].strip("('").strip("')")))):
            t1_P_lightverbDT=t1_P_lightverbDT[1:]
            t1_L_lightverbDT=t1_L_lightverbDT[1:]
            flag_found=1
    if len(t1_P_lightverbDT)>2 and len(t1_L_lightverbDT)>2:
        if (t1_L_lightverbDT[0] in all_verb_verb) and (re.search("^(DT)",t1_P_lightverbDT[1].split("', '")[1].strip("('").strip("')")) and re.search("^(NN|VB)",t1_P_lightverbDT[2].split("', '")[1].strip("('").strip("')"))):
            t1_P_lightverbDT=t1_P_lightverbDT[2:]
            t1_L_lightverbDT=t1_L_lightverbDT[2:]
            flag_found=1
    
    return t1_P_lightverbDT,t1_L_lightverbDT,flag_found

def check_superlative(t1_P_rule,t1_L_rule):
    len_in=len(t1_P_rule)
    t1_P_new=[]
    t1_L_new=[]
    for i in range(0,len_in):
        if (t1_L_rule[i]=="more" or t1_L_rule[i]=="most") and i+1<len_in:
            pos_in=t1_P_rule[i+1].split("', '")[1].strip("('").strip("')")
            if re.search("JJ",pos_in) or re.search("RB",pos_in):
                continue
        t1_P_new.append(t1_P_rule[i])
        t1_L_new.append(t1_L_rule[i])
    return t1_P_new,t1_L_new

def check_JJNN(t1_P_rule_JJNN,t1_L_rule_JJNN):
    len_in=len(t1_P_rule_JJNN)
    t1_P_new=[]
    t1_L_new=[]
    JJ_L_rem=[]
    JJ_P_rem=[]
    for i in range(0,len_in):
        pos_in=t1_P_rule_JJNN[i].split("', '")[1].strip("('").strip("')")
        if re.search("JJ",pos_in) and i+1<len_in:
            pos_in=t1_P_rule_JJNN[i+1].split("', '")[1].strip("('").strip("')")
            if not re.search("NN",pos_in) and not re.search("JJ",pos_in):
                t1_P_new.append(t1_P_rule_JJNN[i])
                t1_L_new.append(t1_L_rule_JJNN[i])
            else:
                JJ_L_rem.append(t1_L_rule_JJNN[i])
                JJ_P_rem.append(t1_P_rule_JJNN[i])
            
        else:
            t1_P_new.append(t1_P_rule_JJNN[i])
            t1_L_new.append(t1_L_rule_JJNN[i])
    return t1_P_new,t1_L_new,JJ_L_rem,JJ_P_rem

def check_gerund_infi(t1_P_rule_JJNN,t1_L_rule_JJNN,L_gerund_infi_to):
    len_in=len(t1_P_rule_JJNN)
    t1_P_new=[]
    t1_L_new=[]
    i=0
    check_gerund_infi_found=0
    while i<len_in:
        if t1_L_rule_JJNN[i] in L_gerund_infi_to and i+2<len_in:
            pos_in=t1_P_rule_JJNN[i+2].split("', '")[1].strip("('").strip("')")
            t1_P_new.append(t1_P_rule_JJNN[i])
            t1_L_new.append(t1_L_rule_JJNN[i])
            if re.search("^(NN|VB)",pos_in) and re.search("to",t1_L_rule_JJNN[i+1]):
                i+=1
                check_gerund_infi_found=1
        else:
            t1_P_new.append(t1_P_rule_JJNN[i])
            t1_L_new.append(t1_L_rule_JJNN[i])
        i+=1
    return t1_P_new,t1_L_new,check_gerund_infi_found
                
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

def strip_DT(t_P,t_L_rule):
    pos_ele=t_P[0].split("', '")[1].strip("('").strip("')")
    dt_list=["a","an","the"]
    '''
    NOTE:
    The determiner word class also includes words that traditional grammars used to classify as adjectives:

    this and that              this dog, that dog
    all, every and some        all dogs, every dog, some dogs
    numbers                    one dog, two dogs, three dogs
    his, her, my etc.          his dog, her dog, my dog, their dog

    Link: http://www.phon.ucl.ac.uk/home/dick/tta/wc/determiners.htm
    '''
    if len(t_P)==1:
        if re.search("^DT",pos_ele) and t_L_rule[0] in dt_list:
            return ([],[])
    if re.search("^DT",pos_ele) and t_L_rule[0] in dt_list:
        t_P=t_P[1:]
        t_L_rule=t_L_rule[1:]
    #if re.search("0;be much of@R@;be a of@R@",line):
    #    print "catch"
    if t_P and t_L_rule:
        pos_ele=t_P[len(t_P)-1].split("', '")[1].strip("('").strip("')")
        if re.search("^DT",pos_ele) and t_L_rule[len(t_P)-1] in dt_list:
            t_P=t_P[:-1]
            t_L_rule=t_L_rule[:-1]
    return (t_P,t_L_rule)

def strip_DT_deep(t_P,t_L_rule):
    pos_ele=t_P[0].split("', '")[1].strip("('").strip("')")
    '''
    NOTE:
    The determiner word class also includes words that traditional grammars used to classify as adjectives:

    this and that              this dog, that dog
    all, every and some        all dogs, every dog, some dogs
    numbers                    one dog, two dogs, three dogs
    his, her, my etc.          his dog, her dog, my dog, their dog

    Link: http://www.phon.ucl.ac.uk/home/dick/tta/wc/determiners.htm
    '''
    if len(t_P)==1:
        if re.search("^DT",pos_ele):# and t_L_rule[0] in dt_list:
            return ([],[])
    if re.search("^DT",pos_ele):# and t_L_rule[0] in dt_list:
        t_P=t_P[1:]
        t_L_rule=t_L_rule[1:]
    #if re.search("0;be much of@R@;be a of@R@",line):
    #    print "catch"
    if t_P and t_L_rule:
        pos_ele=t_P[len(t_P)-1].split("', '")[1].strip("('").strip("')")
        if re.search("^DT",pos_ele):
            t_P=t_P[:-1]
            t_L_rule=t_L_rule[:-1]
    return (t_P,t_L_rule)


def get_derivation(in_word,in_word_P):
    derivation_L=[]
    if flag_derivation:
        if flag_pos_match and len(in_word.split())==1:
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
                if 1:#ele_1.name()==in_word:#ele_1.name().startswith(in_word[0]):
                    derivation_L.append(ele_1.name().replace("_"," "))
                    for ele_2 in ele_1.derivationally_related_forms():
                        derivation_L.append(ele_2.name().replace("_"," "))
        derivation_L.append(in_word.replace("_"," "))
        derivation_L=list(set(derivation_L))
    else:
        derivation_L.append(in_word)
    
    return(derivation_L)
    
def drop_preposition(t1_L_rule,t2_L_rule,t1_P_rule,t2_P_rule):
    t1_L_rule=t1_L_rule[:-1]
    t2_L_rule=t2_L_rule[:-1]
    t1_P_rule=t1_P_rule[:-1]
    t2_P_rule=t2_P_rule[:-1] 
    return t1_L_rule,t2_L_rule,t1_P_rule,t2_P_rule

def rule_active_passive_be(direction,t1_L_rule_ap,t2_L_rule_ap,t1_P_rule_ap,t2_P_rule_ap,del1,flag_positive,prev_dir):
    #if re.search("help protect",line):
    #    print "catch"
    if t1_L_rule_ap and t2_L_rule_ap:
        #print "rev::::",t1_L_rule_ap[len(t1_L_rule_ap)-1]
        flag_beby=0
        flag_ofby=0
        if t1_L_rule_ap[0]=="be" and t2_L_rule_ap[0]!="be" and t1_L_rule_ap[len(t1_L_rule_ap)-1].strip()=="by":
            flag_beby=1
        elif t1_L_rule_ap[len(t1_L_rule_ap)-1].strip()=="by" and t2_L_rule_ap[len(t2_L_rule_ap)-1].strip()=="of":
            flag_ofby=1
        if flag_beby or flag_ofby:
            if flag_beby:
                t1_L_rule_ap=t1_L_rule_ap[1:]
                t1_P_rule_ap=t1_P_rule_ap[1:]
            if flag_ofby:
                t2_L_rule_ap=t1_L_rule_ap[:-1]
                t2_P_rule_ap=t1_P_rule_ap[:-1]
            t1_L_rule_ap=t1_L_rule_ap[:-1]
            t1_P_rule_ap=t1_P_rule_ap[:-1]
            if t2_L_rule_ap and t1_L_rule_ap:
                #if re.search("move towards",line):
                #    print "catch!"
                derivation_t2=get_derivation(" ".join(t2_L_rule_ap),t1_P_rule_ap)
                derivation_t1=get_derivation(" ".join(t1_L_rule_ap),t1_P_rule_ap)
                for word_derivation in derivation_t2:
                    if direction=="fwd":
                        del1,flag_positive,prev_dir=check_thesaurus(t1_P_rule_ap," ".join(t1_L_rule_ap), word_derivation,t2_P_rule_ap, 1, flag_positive, prev_dir,f_w_rule_t1_in_t2,data_w,"fwd",1)
                        del1,flag_positive,prev_dir=check_thesaurus(t2_P_rule_ap,word_derivation," ".join(t1_L_rule_ap),t1_P_rule_ap, 2, flag_positive, prev_dir,f_w_rule_t2_in_t1,data_w,"rev",1)
                    else:
                        del1,flag_positive,prev_dir=check_thesaurus(t1_P_rule_ap," ".join(t1_L_rule_ap), word_derivation,t2_P_rule_ap, 1, flag_positive, prev_dir,f_w_rule_t2_in_t1,data_w,"rev",1)
                        del1,flag_positive,prev_dir=check_thesaurus(t2_P_rule_ap,word_derivation," ".join(t1_L_rule_ap),t1_P_rule_ap, 2, flag_positive, prev_dir,f_w_rule_t1_in_t2,data_w,"fwd",1)
                if not flag_positive:
                    for word_derivation in derivation_t1:
                        if direction=="fwd":
                            del1,flag_positive,prev_dir=check_thesaurus(t2_P_rule_ap," ".join(t2_L_rule_ap), word_derivation,t1_P_rule_ap, 1, flag_positive, prev_dir,f_w_rule_t2_in_t1,data_w,"rev",1)
                            del1,flag_positive,prev_dir=check_thesaurus(t1_P_rule_ap,word_derivation, " ".join(t2_L_rule_ap),t2_P_rule_ap, 2, flag_positive, prev_dir,f_w_rule_t1_in_t2,data_w,"fwd",1)
                        else:
                            #if re.search("move towards",line):
                            #    print "catch"
                            del1,flag_positive,prev_dir=check_thesaurus(t2_P_rule_ap," ".join(t2_L_rule_ap), word_derivation,t1_P_rule_ap, 1, flag_positive, prev_dir,f_w_rule_t1_in_t2,data_w,"fwd",1)
                            del1,flag_positive,prev_dir=check_thesaurus(t1_P_rule_ap,word_derivation, " ".join(t2_L_rule_ap),t2_P_rule_ap, 2, flag_positive, prev_dir,f_w_rule_t2_in_t1,data_w,"rev",1)           
            
    return t1_L_rule_ap,t2_L_rule_ap,t1_P_rule_ap,t2_P_rule_ap,del1,flag_positive,prev_dir

if __name__ == '__main__':
    f=open(path+rule_in_file)
    lines=f.readlines()
    f.close()
    
    f_score=open(path+"out_files\\score.txt","w")
    f_w=open(path+"out_files\\200_tncf_annotated_part2_features_in.csv","w")           
    f_w_not_handled_1=open(path+"out_files\\not_handled_1.csv","w")
    f_w_rule_t2_in_t1=open(path+"out_files\\f_w_rule_t2_in_t1.csv","w")
    f_w_rule_t1_in_t2=open(path+"out_files\\f_w_rule_t1_in_t2.csv","w")
    f_w_whole_t2_in_t1=open(path+"out_files\\whole_t2_in_t1.csv","w")
    f_w_whole_t1_in_t2=open(path+"out_files\\whole_t1_in_t2.csv","w")
    f_w_affix_t2_in_t1=open(path+"out_files\\affix_t2_in_t1.csv","w")
    f_w_affix_t1_in_t2=open(path+"out_files\\affix_t1_in_t2.csv","w")
#     
#     f_score=open(path+"out_files/score.txt","w")
#     f_w=open(path+"out_files/200_tncf_annotated_part2_features_in.csv","w")           
#     f_w_not_handled_1=open(path+"out_files/not_handled_1.csv","w")
#     f_w_rule_t2_in_t1=open(path+"out_files/f_w_rule_t2_in_t1.csv","w")
#     f_w_rule_t1_in_t2=open(path+"out_files/f_w_rule_t1_in_t2.csv","w")
#     f_w_whole_t2_in_t1=open(path+"out_files/whole_t2_in_t1.csv","w")
#     f_w_whole_t1_in_t2=open(path+"out_files/whole_t1_in_t2.csv","w")
#     f_w_affix_t2_in_t1=open(path+"out_files/affix_t2_in_t1.csv","w")
#     f_w_affix_t1_in_t2=open(path+"out_files/affix_t1_in_t2.csv","w")
    f_log=open("log.txt","w")
    
    ###Redundant preposition data###
    f_prep_null=open(path+redundant_prep_file,"r")
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
    
    ###Preposition Synonyms data###
    f_prep=open(path+prep_syn_file)
    freq_prep={}
    for line_prep in f_prep:
        L_line_prep=line_prep.strip("\n").split(";")
        if int(L_line_prep[1])>20:
            freq_prep[L_line_prep[0]]=L_line_prep[1]
    f_prep.close()
        
    stanford_pos_dict={}
    if flag_stanford_pos:
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
        all_tok_pos=st.tag_sents(all_tok)
        for all_tok_i in range(0,len(all_tok)):
            all_tok_str=" ".join(all_tok[all_tok_i])
            stanford_pos_dict[all_tok_str]=all_tok_pos[all_tok_i]
    
    for line in lines:
        flag_positive=0
        flag_affix_call=0
        line=line.strip("\n").replace("'","")
        L_line=line.split(";")
        tag=L_line[0].strip('"')
        if tag=="1":
            all_p=all_p+1
        if tag=="2":
            all_p_2=all_p_2+1
            
        rev_R=L_line[1].endswith("@R@")
        
        rev_noR=L_line[2].split("//")[0].strip("\n").endswith("@R@")
        line_w=re.sub('";','",',line)
#        if re.search("occur as@R@;develop as@R@",line_w):
#            print "found"
        if flag_rev:
            flag_rev_gate=(rev_R and rev_noR) or (not(rev_R) and not(rev_noR))
        else:
            flag_rev_gate=1
        
        t1=L_line[1].strip("@R@").replace("dont ","do not ").replace(" dont"," do not").replace("didnt ","do not ").replace(" didnt"," do not")
        t2=L_line[2].split("//")[0].strip("\n").strip("@R@").replace("dont ","do not ").replace(" dont"," do not").replace("didnt ","do not ").replace(" didnt"," do not")
        t1_L=t1.split()
        t2_L=t2.split()
                    
        if 1:#flag_pos:
            if flag_stanford_pos:
                t1_P=stanford_pos_dict[" ".join(t1_L)]#st.tag(t1_L)#nltk.pos_tag(t1_L)
                t2_P=stanford_pos_dict[" ".join(t2_L)]#st.tag(t2_L)#nltk.pos_tag(t2_L)
                t1_P=[(str(ele[0]),str(ele[1])) for ele in t1_P]
                t2_P=[(str(ele[0]),str(ele[1])) for ele in t2_P]
            else:
                t1_P=nltk.pos_tag(t1_L)
                t2_P=nltk.pos_tag(t2_L)
                
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

                
            f_w.write(line_w)
            t1_P_w=[str(ele).replace("', '", "'; '") for ele in t1_P]
            t2_P_w=[str(ele).replace("', '", "'; '")  for ele in t2_P]
            t1_P=[str(ele) for ele in t1_P]
            t2_P=[str(ele) for ele in t2_P]
            f_w.write(" , "+";".join(t1_P_w)+";".join(t2_P_w)+" , ")
        
        data_w=line+" , "+";".join(t1_P)+";".join(t2_P)+"\n"
        
        #if re.search("email",line):
        #    print ""
            
        if flag_be_trail:
            if re.search(" be$",t1) or re.search(" be$",t2):
                continue
            
        #if re.search("look good to@R@;be good to@R@",line):
        #    print "s"
        if rules_dt:#flag_dt:
            t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L, t2_L,t1_P,t2_P)
            #t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L_rule, t2_L_rule,t1_P_rule,t2_P_rule)
            if 1:#rules_dt:
                if len(t1_affix)>0:
                    t1_P_affix,t1_affix=strip_DT(t1_P_affix,t1_affix)
                if len(t2_affix):
                    t2_P_affix,t2_affix=strip_DT(t2_P_affix,t2_affix)
            
            if not (t1_affix) and not(t2_affix):
                flag_positive=1
                f_w_affix_t1_in_t2.write(line_w+"\n")
            
            
        if flag_whole and flag_rev_gate:
            prev_dir=0
            flag_derivation=0
            #flag_multiple_word_back=flag_multiple_word
            #flag_multiple_word=0
            del1,flag_positive,prev_dir=check_thesaurus(t1_P,t1, t2,t2_P, 1, flag_positive, prev_dir,f_w_whole_t2_in_t1,data_w,"fwd",1)
            f_w.write(del1+" , ")
            del1,flag_positive,prev_dir=check_thesaurus(t2_P,t2, t1,t1_P, 2, flag_positive, prev_dir,f_w_whole_t1_in_t2,data_w,"rev",1)
            f_w.write(del1+" , ")
            if not flag_positive and not (re.search(t1+"$",t2) or re.search(t2+"$",t1)) and not re.search("^be", t1) and not re.search("^be", t2):
                tmp_list_t1=get_synonyms([t1])
                if len(tmp_list_t1)>0 and len(t1.split()) > 1:
                    #print line
                    f_log.write("t1 -- >"+line+"\n")
                    if tag=="1":
                        f_w_not_handled_1.write(line+" , "+";".join(t1_P)+";".join(t2_P)+"\n")
                    continue
                tmp_list_t2=get_synonyms([t2])
                if len(tmp_list_t2)>0 and len(t2.split()) > 1:
                    #print line
                    f_log.write("t2 -- >"+line+"\n")
                    if tag=="1":
                        f_w_not_handled_1.write(line+" , "+";".join(t1_P)+";".join(t2_P)+"\n")
                    continue
            flag_derivation=1
            #flag_multiple_word=flag_multiple_word_back
        
        #Handling NOT
        if flag_not and flag_affix and not flag_positive:
            prev_dir=0
            #global flag_affix_call
            flag_affix_call=1
            t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L, t2_L,t1_P,t2_P)
            #t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L_rule, t2_L_rule,t1_P_rule,t2_P_rule)
            if rules_dt:
                #print t1_P_affix,t1_affix
                if len(t1_affix)>0:
                    t1_P_affix,t1_affix=strip_DT(t1_P_affix,t1_affix)
                #print t2_P_affix,t2_affix
                if len(t2_affix):
                    t2_P_affix,t2_affix=strip_DT(t2_P_affix,t2_affix)
            
            #print "\n",t1,";",t2,"\nAffix : ",t1_affix,";",t2_affix
            if (not t1_affix and t2_affix=="not") or(not t2_affix and t1_affix=="not"):
                flag_affix_call=0
                continue 
            flag_affix_call=0

        #if re.search("look good to",line):
        #    print ""
                
        if flag_rules:
            #if re.search("love to have;love have",line):
            #    print "catch"
            t1_L_rule=t1_L
            t2_L_rule=t2_L
            t1_P_rule=t1_P
            t2_P_rule=t2_P
            rule_have_fire=0
            
            ###Gerund-Infinitive Equivalence Rule###
            if flag_gerund_infi_to:
                #if re.search("start to have@R@;have",line):
                #    print "d"
                t1_P_rule,t1_L_rule,check_gerund_infi_found=check_gerund_infi(t1_P_rule,t1_L_rule,L_gerund_infi_to)
                t2_P_rule,t2_L_rule,check_gerund_infi_found=check_gerund_infi(t2_P_rule,t2_L_rule,L_gerund_infi_to)
                
                #if re.search("want to talk about@R@;like to talk about",line):
                #    print "catch"
                t1_P_rule,t1_L_rule,check_gerund_infi_found=check_gerund_infi(t1_P_rule,t1_L_rule,L_gerund_infi_to_t1)
                if not flag_positive: 
                    t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L_rule, t2_L_rule,t1_P_rule,t2_P_rule)
                    if not(t1_affix) and not(t2_affix):
                        flag_positive=1
                        f_w_rule_t2_in_t1.write(data_w)
                    
            if rule_superlative:#more most
                #if re.search("be an integral part of@R@;be the most important part of@R@",line):
                #    print ""
                t1_P_rule,t1_L_rule=check_superlative(t1_P_rule,t1_L_rule)
                t2_P_rule,t2_L_rule=check_superlative(t2_P_rule,t2_L_rule)
                
            if rule_JJ_NN_mod:
                #if re.search("offer a full range of@R@;provide a wide range of@R@",line):
                #    print "Dd"
                t1_P_rule_back,t1_L_rule_back,t1_JJ_L_rem,t1_JJ_P_rem=check_JJNN(t1_P_rule,t1_L_rule)
                t2_P_rule_back,t2_L_rule_back,t2_JJ_L_rem,t2_JJ_P_rem=check_JJNN(t2_P_rule,t2_L_rule)
                if t1_JJ_L_rem and t2_JJ_L_rem:
                    JJ_flag_positive=0
                    JJ_flag_positive_back=1
                    for temp_i in range(0,len(t1_JJ_L_rem)):#last JJ match
                        del1,JJ_flag_positive,prev_dir=check_thesaurus([t1_JJ_P_rem[temp_i]],[t1_JJ_L_rem[temp_i]],[t2_JJ_L_rem[temp_i]],[t2_JJ_P_rem[temp_i]],1,0,prev_dir,f_w_rule_t1_in_t2,"","fwd",0)
                        del1,JJ_flag_positive,prev_dir=check_thesaurus([t2_JJ_P_rem[temp_i]],[t2_JJ_L_rem[temp_i]],[t1_JJ_L_rem[temp_i]],[t1_JJ_P_rem[temp_i]],2,JJ_flag_positive,prev_dir,f_w_rule_t1_in_t2,"","rev",0)
                        JJ_flag_positive_back=JJ_flag_positive_back and JJ_flag_positive
                    JJ_flag_positive=JJ_flag_positive_back
                        
                    if JJ_flag_positive:
                        f_log.write(line_w)
                        t2_P_rule=t2_P_rule_back
                        t2_L_rule=t2_L_rule_back
                        t1_P_rule=t1_P_rule_back
                        t1_L_rule=t1_L_rule_back
                        
                        #flag_affix_call=1
                        prev_dir=0
                        t1_affix_rule_back,t2_affix_rule_back,t1_P_affix_rule_back,t2_P_affix_rule_back=remove_common_affix(t1_L_rule_back, t2_L_rule_back,t1_P_rule_back,t2_P_rule_back)
                        
                        if rules_dt:
                            #print t1_P_affix,t1_affix
                            if len(t1_affix_rule_back)>0:
                                t1_P_affix_rule_back,t1_affix_rule_back=strip_DT(t1_P_affix_rule_back,t1_affix_rule_back)
                            #print t2_P_affix,t2_affix
                            if len(t2_affix_rule_back):
                                t2_P_affix_rule_back,t2_affix_rule_back=strip_DT(t2_P_affix_rule_back,t2_affix_rule_back)
                        
                        #print "\n",t1,";",t2,"\nAffix : ",t1_affix,";",t2_affix
                        t1_affix_rule_back=" ".join(t1_affix_rule_back)
                        t2_affix_rule_back=" ".join(t2_affix_rule_back)
                        del1,flag_positive,prev_dir=check_thesaurus(t1_P_affix_rule_back,t1_affix_rule_back, t2_affix_rule_back,t2_P_affix_rule_back, 1, flag_positive, prev_dir,f_w_affix_t2_in_t1,data_w,"fwd",1)
                        f_w.write(del1+" , ")
                        del1,flag_positive,prev_dir=check_thesaurus(t2_P_affix_rule_back,t2_affix_rule_back, t1_affix_rule_back,t1_P_affix_rule_back, 2, flag_positive, prev_dir,f_w_affix_t1_in_t2,data_w,"rev",1)
                        f_w.write(del1+" , ") 
                elif t1_L_rule_back:
                    t2_P_rule=t2_P_rule_back
                    t2_L_rule=t2_L_rule_back
                    t1_P_rule=t1_P_rule_back
                    t1_L_rule=t1_L_rule_back
                    
                if not flag_positive: 
                    t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L_rule, t2_L_rule,t1_P_rule,t2_P_rule)
                    if not(t1_affix) and not(t2_affix):
                        flag_positive=1
                        f_w_rule_t2_in_t1.write(data_w)
                        
                        
            if rule_lightverb_dt:
                #take, have, give, do, make,
                t1_P_rule_lightverbDT=t1_P_rule
                t2_P_rule_lightverbDT=t2_P_rule
                t1_L_rule_lightverbDT=t1_L_rule
                t2_L_rule_lightverbDT=t2_L_rule
                flag_found=0
                #if re.search("changed by",line):
                #    print "catch"
                #if re.search("help protect",line):
                #    print "catch"
                
                t1_P_rule,t1_L_rule,flag_found=check_lightverb_construction(t1_P_rule_lightverbDT,t1_L_rule_lightverbDT)
                if not flag_positive: 
                    t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L_rule, t2_L_rule,t1_P_rule,t2_P_rule)
                    if not(t1_affix) and not(t2_affix):
                        flag_positive=1
                        f_w_rule_t2_in_t1.write(data_w)
                    
            if rule_have_JJ:#Not needed#Never runs
                #"1";have a huge impact on@R@;impact on@R@::have impact on
                t1_P_rule_haveJJ=t1_P_rule
                t2_P_rule_haveJJ=t2_P_rule
                t1_L_rule_haveJJ=t1_L_rule
                t2_L_rule_haveJJ=t2_L_rule
                
                t1_P_rule_haveJJ_strip=[]
                t2_P_rule_haveJJ_strip=[]
                t1_L_rule_haveJJ_strip=[]
                t2_L_rule_haveJJ_strip=[]
                have_JJ_update=0
                if t1_L_rule_haveJJ[0]=="have":
                    i_rule=0
                    for ele_P in t1_P_rule_haveJJ[1:]:
                        i_rule=i_rule+1
                        pos=ele_P.split("', ")[1].strip("')")
                        if pos=="DT":
                            continue
                        elif pos=="JJ":
                            for i in range(0,len(t1_P_rule_haveJJ)):
                                if i>=1 and i<=i_rule:
                                    continue
                                t1_L_rule_haveJJ_strip.append(t1_L_rule_haveJJ[i])
                                t1_P_rule_haveJJ_strip.append(t1_P_rule_haveJJ[i])
                                have_JJ_update=1
                            break
                        else:
                            break
                
                if have_JJ_update:
                    t1_P_rule=t1_P_rule_haveJJ_strip
                    t1_L_rule=t1_L_rule_haveJJ_strip
                    #print "have JJ",t1_L_rule,t2_L_rule,line
                have_JJ_update=0
                if t2_L_rule_haveJJ[0]=="have":
                    i_rule=0
                    for ele_P in t2_P_rule_haveJJ[1:]:
                        i_rule=i_rule+1
                        pos=ele_P.split("', ")[1].strip("')")
                        if pos=="DT":
                            continue
                        elif pos=="JJ":
                            for i in range(0,len(t2_P_rule_haveJJ)):
                                if i>=1 and i<=i_rule:
                                    continue
                                t2_L_rule_haveJJ_strip.append(t2_L_rule_haveJJ[i])
                                t2_P_rule_haveJJ_strip.append(t2_P_rule_haveJJ[i])
                                have_JJ_update=1
                            break
                        else:
                            break
                if have_JJ_update:
                    t2_P_rule=t2_P_rule_haveJJ_strip
                    t2_L_rule=t2_L_rule_haveJJ_strip
                    #print "have JJ",t1_L_rule,t2_L_rule,line
                have_JJ_update=0
                
            if rule_have:
                #"1";have responsibility for;be accountable for
                if t1_L_rule[0]=="have":
                    pos_ele_2=t2_P_rule[0].split("', '")[1].strip("('").strip("')")
                    if t2_L_rule[0]=="be":#re.search("^VB",pos_ele_2):
                        #print "Rule:"
                        #print t1,t2
                        t1_L_rule=t1_L_rule[1:]
                        t2_L_rule=t2_L_rule[1:]
                        t1_P_rule=t1_P_rule[1:]
                        t2_P_rule=t2_P_rule[1:]
                        prev_dir=0
                        del1,flag_positive,prev_dir=check_thesaurus(t1_P_rule,t1_L_rule, t2_L_rule,t2_P_rule, 1, flag_positive, prev_dir,f_w_rule_t2_in_t1,data_w,"fwd",1)
                        #print "RULE del",del1
                        del1,flag_positive,prev_dir=check_thesaurus(t2_P_rule,t2_L_rule,t2_P_rule, t1_L_rule, 2, flag_positive, prev_dir,f_w_rule_t1_in_t2,data_w,"rev",1)
                        #print "RULE del",del1
                        rule_have_fire=1
                        if not flag_positive: 
                            t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L_rule, t2_L_rule,t1_P_rule,t2_P_rule)
                            if not(t1_affix) and not(t2_affix):
                                flag_positive=1
                                f_w_rule_t2_in_t1.write(data_w)
                
            if rules_dt:
                t1_P_rule,t1_L_rule=strip_DT(t1_P_rule,t1_L_rule)
                t2_P_rule,t2_L_rule=strip_DT(t2_P_rule,t2_L_rule)
                if not flag_positive: 
                    t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L_rule, t2_L_rule,t1_P_rule,t2_P_rule)
                    if not(t1_affix) and not(t2_affix):
                        flag_positive=1
                        f_w_rule_t2_in_t1.write(data_w)
                
            if rule_be:
                if t1_L_rule[0]=="be" and t2_L_rule[0]=="be":
                    #print "\n",t1_L_rule,t2_L_rule
                    t1_L_rule=t1_L_rule[1:]
                    t1_P_rule=t1_P_rule[1:]   
                    #print t1_L_rule,t2_L_rule
                    t2_L_rule=t2_L_rule[1:]
                    t2_P_rule=t2_P_rule[1:]
                    if not flag_positive: 
                        t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L_rule, t2_L_rule,t1_P_rule,t2_P_rule)
                        if not(t1_affix) and not(t2_affix):
                            flag_positive=1
                            f_w_rule_t2_in_t1.write(data_w)
                    
            if rules_dt:
                t1_P_rule,t1_L_rule=strip_DT(t1_P_rule,t1_L_rule)
                t2_P_rule,t2_L_rule=strip_DT(t2_P_rule,t2_L_rule)
                if not flag_positive: 
                    t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L_rule, t2_L_rule,t1_P_rule,t2_P_rule)
                    if not(t1_affix) and not(t2_affix):
                        flag_positive=1
                        f_w_rule_t2_in_t1.write(data_w)
                
                        
            #print " ".join(t2_L_rule)," ".join(t1_L_rule)
            
            if rule_be_prep:
                #if re.search("be sick of@R@;tire of@R@",line):
                #    print ""
                #if re.search("look more like;sound more like",line):
                #    print ""
                t1_P_rule_ap=t1_P_rule
                t2_P_rule_ap=t2_P_rule
                t1_L_rule_ap=t1_L_rule
                t2_L_rule_ap=t2_L_rule
                if t1_L_rule_ap and t2_L_rule_ap and len(t1_L_rule_ap)>1 and len(t2_L_rule_ap)>1:
                    if t1_L_rule_ap[0]=="be" and t2_L_rule_ap[0]!="be" and t1_L_rule_ap[len(t1_L_rule_ap)-1].strip()==t2_L_rule_ap[len(t2_L_rule_ap)-1].strip() and t1_P_rule_ap[len(t1_P_rule_ap)-1].split(', ')[1].strip("')")=="IN":#"by"
                        t1_L_rule_ap=t1_L_rule_ap[1:]
                        t1_L_rule_ap=t1_L_rule_ap[:-1]
                        t1_P_rule_ap=t1_P_rule_ap[1:]
                        t1_P_rule_ap=t1_P_rule_ap[:-1]
                        t2_L_rule_ap=t2_L_rule_ap[:-1]
                        t2_P_rule_ap=t2_P_rule_ap[:-1]
                        if t2_L_rule_ap and t1_L_rule_ap:
                            derivation_t2=get_derivation(" ".join(t2_L_rule_ap),t2_P_rule_ap)
                            derivation_t2.append(" ".join(t2_L_rule_ap))               
                            derivation_t1=get_derivation(" ".join(t1_L_rule_ap),t1_P_rule_ap)
                            derivation_t1.append(" ".join(t1_L_rule_ap)) 
                        #print thesaurus[]
                            for word_derivation in derivation_t2:
                                #print "active-passive",t1_L_rule,t2_L_rule
                                del1,flag_positive,prev_dir=check_thesaurus(t1_P_rule_ap," ".join(t1_L_rule_ap), word_derivation, t2_P_rule_ap,1, flag_positive, prev_dir,f_w_rule_t2_in_t1,data_w,"fwd",1)
                                del1,flag_positive,prev_dir=check_thesaurus(t2_P_rule_ap,word_derivation," ".join(t1_L_rule_ap),t1_P_rule_ap, 2, flag_positive, prev_dir,f_w_rule_t1_in_t2,data_w,"rev",1)
                            if not flag_positive:
                                for word_derivation in derivation_t1:
                                    #print "active-passive 2",t1_L_rule,t2_L_rule
                                    del1,flag_positive,prev_dir=check_thesaurus(t2_P_rule_ap," ".join(t2_L_rule_ap), word_derivation,t1_P_rule_ap, 1, flag_positive, prev_dir,f_w_rule_t2_in_t1,data_w,"rev",1)
                                    del1,flag_positive,prev_dir=check_thesaurus(t1_P_rule_ap,word_derivation, " ".join(t2_L_rule_ap),t2_P_rule_ap, 2, flag_positive, prev_dir,f_w_rule_t1_in_t2,data_w,"fwd",1)
                    
                        if not flag_positive: 
                            t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L_rule, t2_L_rule,t1_P_rule,t2_P_rule)
                            if not(t1_affix) and not(t2_affix):
                                flag_positive=1
                                f_w_rule_t2_in_t1.write(data_w)
                                            
                                
                    elif t2_L_rule_ap[0]=="be" and t1_L_rule_ap[0]!="be" and t2_L_rule_ap[len(t2_L_rule_ap)-1].strip()==t1_L_rule_ap[len(t1_L_rule_ap)-1].strip() and t2_P_rule_ap[len(t2_P_rule_ap)-1].split(', ')[1].strip("')")=="IN":#"by":
                        t2_L_rule_ap=t2_L_rule_ap[:-1]
                        t2_L_rule_ap=t2_L_rule_ap[1:]
                        t2_P_rule_ap=t2_P_rule_ap[:-1]
                        t2_P_rule_ap=t2_P_rule_ap[1:]
                        t1_L_rule_ap=t1_L_rule_ap[:-1]
                        t1_P_rule_ap=t1_P_rule_ap[:-1]
                        
                        if t2_L_rule_ap and t1_L_rule_ap:
                            derivation_t2=get_derivation(" ".join(t2_L_rule_ap),t2_P_rule_ap)
                            derivation_t2.append(" ".join(t2_L_rule_ap))               
                            derivation_t1=get_derivation(" ".join(t1_L_rule_ap),t1_P_rule_ap)
                            derivation_t1.append(" ".join(t1_L_rule_ap)) 
                            #print thesaurus[]
                            
                            for word_derivation in derivation_t2:
                                #print "active-passive 2",t1_L_rule,t2_L_rule
                                del1,flag_positive,prev_dir=check_thesaurus(t1_P_rule_ap," ".join(t1_L_rule_ap), word_derivation,t2_P_rule_ap, 1, flag_positive, prev_dir,f_w_rule_t2_in_t1,data_w,"fwd",1)
                                del1,flag_positive,prev_dir=check_thesaurus(t2_P_rule_ap,word_derivation, " ".join(t1_L_rule_ap),t1_P_rule_ap, 2, flag_positive, prev_dir,f_w_rule_t1_in_t2,data_w,"rev",1)
         
                            if not flag_positive:
                                for word_derivation in derivation_t1:
                                    #print "active-passive 2",t1_L_rule,t2_L_rule
                                    del1,flag_positive,prev_dir=check_thesaurus(t2_P_rule_ap," ".join(t2_L_rule_ap), word_derivation,t1_P_rule_ap, 1, flag_positive, prev_dir,f_w_rule_t2_in_t1,data_w,"rev",1)
                                    del1,flag_positive,prev_dir=check_thesaurus(t1_P_rule_ap,word_derivation, " ".join(t2_L_rule_ap),t2_P_rule_ap, 2, flag_positive, prev_dir,f_w_rule_t1_in_t2,data_w,"fwd",1)
                        if not flag_positive: 
                            t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L_rule, t2_L_rule,t1_P_rule,t2_P_rule)
                            if not(t1_affix) and not(t2_affix):
                                flag_positive=1
                                f_w_rule_t2_in_t1.write(data_w)
            
            ###Preposition Synonym Rule###                    
            if rule_equal_preposition:
                flag_derivation=0
                equal_preposition_L=[("into","into"),("towards","towards"),("onto","onto"),("from","from"),("till","till"),("until","until"),("across","across"),("through","through"),("along","along"),("around","around"),("up","up"),("down","down"),("over","over"),("under","under"),("as","to be"),("in","at"),("in","by"),("in","to"),("on","on"),("for","for"),("of","of"),("to","to"),("about","about"),("with","with")]
                if rule_equal_preposition_data_freq:
                    for key_prep in freq_prep:
                        L_key_prep=key_prep.split(",")
                        for ele_prep_comb in itertools.combinations(L_key_prep,2):
                                equal_preposition_L.append(ele_prep_comb)

                #if re.search("be located near@R@;be located at@R@",line):
                #    print ""                           
                if not flag_rev_gate:
                    #be-by condition checked in active pasive - so removal of by in rev cases isn't allowed here
                    equal_preposition_L=[equal_preposition_L_ele for equal_preposition_L_ele in equal_preposition_L if (not equal_preposition_L_ele[0]=="by") and (not equal_preposition_L_ele[1]=="by")]
                #http://www.ego4u.com/en/cram-up/grammar/prepositions
                check_prep,t1_L_rule, t2_L_rule,t1_P_rule, t2_P_rule,prep_pair=check_preposition_equivalence(t1_L_rule, t2_L_rule, equal_preposition_L, 0,t1_P_rule,t2_P_rule)
                
                if check_prep:
                    del1,flag_positive,prev_dir=check_thesaurus(t1_P_rule," ".join(t1_L_rule), " ".join(t2_L_rule),t2_P_rule, 1, flag_positive, prev_dir,f_w_rule_t2_in_t1,data_w,"fwd",1)
                    del1,flag_positive,prev_dir=check_thesaurus(t2_P_rule," ".join(t2_L_rule), " ".join(t1_L_rule), t1_P_rule,2, flag_positive, prev_dir,f_w_rule_t1_in_t2,data_w,"rev",1)
                    if not flag_positive and prep_pair==("as","to be"):
                        derivation_t2=get_derivation(" ".join(t2_L_rule),t2_P_rule)
                        derivation_t2.append(" ".join(t2_L_rule))               
                        derivation_t1=get_derivation(" ".join(t1_L_rule),t1_P_rule)
                        derivation_t1.append(" ".join(t1_L_rule)) 
                        #print thesaurus[]
                        for word_derivation in derivation_t2:
                            #print "active-passive",t1_L_rule,t2_L_rule
                            del1,flag_positive,prev_dir=check_thesaurus(t1_P_rule," ".join(t1_L_rule), word_derivation,t2_P_rule, 1, flag_positive, prev_dir,f_w_rule_t2_in_t1,data_w,"fwd",1)
                            del1,flag_positive,prev_dir=check_thesaurus(t2_P_rule,word_derivation," ".join(t1_L_rule),t1_P_rule, 2, flag_positive, prev_dir,f_w_rule_t1_in_t2,data_w,"rev",1)
                        if not flag_positive:
                            for word_derivation in derivation_t1:
                                #print "active-passive 2",t1_L_rule,t2_L_rule
                                del1,flag_positive,prev_dir=check_thesaurus(t2_P_rule," ".join(t2_L_rule), word_derivation,t1_P_rule, 1, flag_positive, prev_dir,f_w_rule_t2_in_t1,data_w,"rev",1)
                                del1,flag_positive,prev_dir=check_thesaurus(t1_P_rule,word_derivation, " ".join(t2_L_rule),t2_P_rule, 2, flag_positive, prev_dir,f_w_rule_t1_in_t2,data_w,"fwd",1)
                    if not flag_positive: 
                        t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L_rule, t2_L_rule,t1_P_rule,t2_P_rule)
                        if not(t1_affix) and not(t2_affix):
                            flag_positive=1
                            f_w_rule_t2_in_t1.write(data_w)
                
                flag_derivation=1
                
            if rule_have_fire and (t1_L_rule or t2_L_rule):
                t1_a_rule,t2_a_rule,t1_ap_rule,t2_ap_rule=remove_common_affix(t1_L_rule,t2_L_rule,t1_P_rule,t2_P_rule)
                if t1_a_rule and t2_a_rule:
                    derivation_t2=get_derivation(" ".join(t2_a_rule),t2_ap_rule)
                    derivation_t2.append(" ".join(t2_a_rule))               
                    derivation_t1=get_derivation(" ".join(t1_a_rule),t1_ap_rule)
                    derivation_t1.append(" ".join(t1_a_rule)) 
                    #print thesaurus[]
                    for word_derivation in derivation_t2:
                        #if len(t1_a_rule)>1 or len(t2_a_rule)>1:
                            #print "ISSUE",t1_L_rule,t2_L_rule
                            #print "ISSUE",t1_a_rule,t2_a_rule
                            #print word_derivation
                        #if re.search("have until;be until",line):
                        #    print "caught!"
                        del1,flag_positive,prev_dir=check_thesaurus(t1_ap_rule,t1_a_rule[0], word_derivation,t2_ap_rule, 1, flag_positive, prev_dir,f_w_rule_t2_in_t1,data_w,"fwd",1)
                        del1,flag_positive,prev_dir=check_thesaurus(t2_ap_rule,word_derivation, t1_a_rule[0],t1_ap_rule, 2, flag_positive, prev_dir,f_w_rule_t1_in_t2,data_w,"rev",1)
                    
                    if not flag_positive:
                        for word_derivation in derivation_t1:
                            #print "active-passive 2",t1_L_rule,t2_L_rule
                            del1,flag_positive,prev_dir=check_thesaurus(t1_ap_rule,word_derivation,t2_a_rule[0],t2_ap_rule, 1, flag_positive, prev_dir,f_w_rule_t2_in_t1,data_w,"fwd",1)
                            del1,flag_positive,prev_dir=check_thesaurus(t2_ap_rule,t2_a_rule[0],word_derivation,t1_ap_rule, 2, flag_positive, prev_dir,f_w_rule_t1_in_t2,data_w,"rev",1)
                                
            if not flag_rev_gate and rule_active_passive_be:
                t1_P_rule_ap=t1_P_rule
                t2_P_rule_ap=t2_P_rule
                t1_L_rule_ap=t1_L_rule
                t2_L_rule_ap=t2_L_rule
                prev_dir=0
                del1="False"
                t1_L_rule_ap,t2_L_rule_ap,t1_P_rule_ap,t2_P_rule_ap,del1,flag_positive,prev_dir= rule_active_passive_be("fwd",t1_L_rule_ap, t2_L_rule_ap, t1_P_rule_ap, t2_P_rule_ap, del1, flag_positive, prev_dir)
                t2_L_rule_ap, t1_L_rule_ap, t2_P_rule_ap, t1_P_rule_ap, del1, flag_positive, prev_dir=rule_active_passive_be("rev",t2_L_rule_ap, t1_L_rule_ap, t2_P_rule_ap, t1_P_rule_ap, del1, flag_positive, prev_dir)
                
            if not flag_positive:
                t1_P_rule,t1_L_rule,flag_found=check_verbverb_construction(t1_P_rule_lightverbDT,t1_L_rule_lightverbDT,t2_L_rule_lightverbDT)
            if not flag_positive:
                prev_dir=1
                if t1_L_rule==t2_L_rule:
                    del1,flag_positive,prev_dir=check_thesaurus(t1_P_rule,t1_L_rule,t2_L_rule,t2_P_rule, 2, flag_positive, prev_dir,f_w_rule_t2_in_t1,data_w,"fwd",1)           
                
        if not flag_positive and flag_wordnet_hypernym_hyponym:#regret@R@;feel@R@
            #if re.search("0;sit to;follow to",line):
            #    print ""
            prev_dir=0
            flag_affix_call=1
            t1_affix_wn,t2_affix_wn,t1_P_affix_wn,t2_P_affix_wn=remove_common_affix(t1_L, t2_L,t1_P,t2_P)
            
            if rules_dt:
                if len(t1_affix_wn)>0:
                    t1_P_affix_wn,t1_affix_wn=strip_DT(t1_P_affix_wn,t1_affix_wn)
                if len(t2_affix_wn):
                    t2_P_affix_wn,t2_affix_wn=strip_DT(t2_P_affix_wn,t2_affix_wn)
            if t1_affix_wn and t2_affix_wn:
                #print "\n",t1,";",t2,"\nAffix : ",t1_affix_wn,";",t2_affix_wn
                t1_affix_wn="-".join(t1_affix_wn)
                t2_affix_wn="-".join(t2_affix_wn)
                flag_hypernym, flag_hyponym,flag_common_hypernym=check_relation(t1_P_affix_wn,t1_affix_wn, t2_affix_wn,t2_P_affix_wn, 1, flag_positive, prev_dir,f_w_affix_t2_in_t1,data_w,"fwd",1)
                if flag_hypernym or flag_hyponym or flag_common_hypernym:
                    f_w_rule_t2_in_t1.write(data_w)
                    flag_positive=1
                f_w.write(del1+" , ")
                f_w.write(del1+" , ")
                flag_affix_call=0    
        
        if not flag_positive and flag_affix:
            #if re.search("spend some time in;spend time in",line):
            #    print ""
            prev_dir=0
            #global flag_affix_call
            flag_affix_call=1
            t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L, t2_L,t1_P,t2_P)
            if rules_dt:
                if len(t1_affix)>0:
                    t1_P_affix,t1_affix=strip_DT(t1_P_affix,t1_affix)
                if len(t2_affix):
                    t2_P_affix,t2_affix=strip_DT(t2_P_affix,t2_affix)
            
            if not (t1_affix) and not(t2_affix):
                flag_positive=1
                f_w_affix_t1_in_t2.write(line_w+"\n")
            else:
                #print "\n",t1,";",t2,"\nAffix : ",t1_affix,";",t2_affix
                t1_affix=" ".join(t1_affix)
                t2_affix=" ".join(t2_affix)
                del1,flag_positive,prev_dir=check_thesaurus(t1_P_affix,t1_affix, t2_affix,t2_P_affix, 1, flag_positive, prev_dir,f_w_affix_t2_in_t1,data_w,"fwd",1)
                f_w.write(del1+" , ")
                del1,flag_positive,prev_dir=check_thesaurus(t2_P_affix,t2_affix, t1_affix,t1_P_affix, 2, flag_positive, prev_dir,f_w_affix_t1_in_t2,data_w,"rev",1)
                f_w.write(del1+" , ")
            flag_affix_call=0

        
        if not flag_positive and flag_affix and rule_preposition_null:#redundant preposition
            prev_dir=0
            #global flag_affix_call
            flag_affix_call=1
            t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L, t2_L,t1_P,t2_P)
            if rules_dt:
                if len(t1_affix)>0:
                    t1_P_affix,t1_affix=strip_DT_deep(t1_P_affix,t1_affix)
                if len(t2_affix):
                    t2_P_affix,t2_affix=strip_DT(t2_P_affix,t2_affix)
            
            if not (t1_affix) and not(t2_affix):
                flag_positive=1
                f_w_affix_t1_in_t2.write(line_w+"\n")
            else:
                #print "\n",t1,";",t2,"\nAffix : ",t1_affix,";",t2_affix
                t1_affix=" ".join(t1_affix)
                t2_affix=" ".join(t2_affix)
            
                t1_affix_pn=[]
                t2_affix_pn=[]
                
                t1_affix_pn_P=[]
                t2_affix_pn_P=[]
                
                if rule_preposition_null and t1_affix and t2_affix:
                    #if re.search("be the start of",line_w):
                    #    print "ss"
                    #print t1_affix
                    #print t2_affix
                    #print t1_P_affix
                    #print t2_P_affix
                    temp_a=t1_affix.split()
                    i_temp_a=0

                    while i_temp_a<len(temp_a):
                        ele_a=temp_a[i_temp_a]
                        if i_temp_a+1<len(temp_a) and ele_a in prep_null.keys():
                            if temp_a[i_temp_a+1] in prep_null[ele_a]:
                                t1_affix_pn.append(ele_a)
                                t1_affix_pn_P.append(t1_P_affix[i_temp_a])
                                i_temp_a+=2
                                continue 
                        t1_affix_pn.append(ele_a)
                        t1_affix_pn_P.append(t1_P_affix[i_temp_a])
                        i_temp_a+=1
                    
                    temp_a=t2_affix.split()
                    i_temp_a=0
                        
                    while i_temp_a<len(temp_a):
                        ele_a=temp_a[i_temp_a]
                        if i_temp_a+1<len(temp_a):
                            if ele_a in prep_null.keys():
                                if temp_a[i_temp_a+1] in prep_null[ele_a]:
                                    t2_affix_pn.append(ele_a)
                                    t2_affix_pn_P.append(t2_P_affix[i_temp_a])
                                    i_temp_a+=2
                                    continue 
                        t2_affix_pn.append(ele_a)
                        t2_affix_pn_P.append(t2_P_affix[i_temp_a])
                        i_temp_a+=1
                    
                    t1_affix=" ".join(t1_affix_pn)
                    t2_affix=" ".join(t2_affix_pn)
                    t1_P_affix=t1_affix_pn_P
                    t2_P_affix=t2_affix_pn_P
                            
                    del1,flag_positive,prev_dir=check_thesaurus(t1_P_affix,t1_affix, t2_affix,t2_P_affix, 1, flag_positive, prev_dir,f_w_affix_t2_in_t1,data_w,"fwd",1)
                    f_w.write(del1+" , ")
                    del1,flag_positive,prev_dir=check_thesaurus(t2_P_affix,t2_affix, t1_affix,t1_P_affix, 2, flag_positive, prev_dir,f_w_affix_t1_in_t2,data_w,"rev",1)
                    f_w.write(del1+" , ")
            flag_affix_call=0
            
        if flag_deep_dt:
            t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L, t2_L,t1_P,t2_P)
            #t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L_rule, t2_L_rule,t1_P_rule,t2_P_rule)
            if rules_dt:
                if len(t1_affix)>0:
                    t1_P_affix,t1_affix=strip_DT_deep(t1_P_affix,t1_affix)
                if len(t2_affix):
                    t2_P_affix,t2_affix=strip_DT(t2_P_affix,t2_affix)
            
            if not (t1_affix) and not(t2_affix) and not flag_positive:
                flag_positive=1
                f_w_affix_t1_in_t2.write(line_w+"\n")
        
        if flag_noun_verb_wn and not flag_positive:
            t1_L_n=t1_L
            t2_L_n=t2_L
            t1_P_n=t1_P
            t2_P_n=t2_P
            if (len(t2_L_n)==1 and t1_L_n[0]=="be" and t1_L[len(t1_L_n)-1]=="of"):
                t1_L_n=t1_L_n[1:]
                t1_L_n=t1_L_n[:-1]  
                t1_P_n=t1_P_n[1:]
                t1_P_n=t1_P_n[:-1]
                if len(t1_L_n)>0:
                    t1_P_n,t1_L_n=strip_DT(t1_P_n,t1_L_n)
                              
                if t1_L_n:
                    if nounify_guided(("").join(t2_L_n), ("").join(t1_L_n)):#verb then noun:
                        flag_positive=1
                        f_w_affix_t1_in_t2.write(line_w+"\n")
                        
            elif (len(t1_L_n)==1 and t2_L_n[0]=="be" and t2_L[len(t2_L_n)-1]=="of"):
                t2_L_n=t2_L_n[1:]
                t2_L_n=t2_L_n[:-1]  
                t2_P_n=t2_P_n[1:]
                t2_P_n=t2_P_n[:-1]
                if len(t2_L_n)>0:
                    t2_P_n,t2_L_n=strip_DT_deep(t2_P_n,t2_L_n)
                              
                if t2_L_n:
                    if nounify_guided(("").join(t1_L_n), ("").join(t2_L_n)):#verb then noun:
                        flag_positive=1
                        f_w_affix_t1_in_t2.write(line_w+"\n")
                            
        f_w.write("\n")
                    
        if flag_positive==0 and tag=="1":
            f_w_not_handled_1.write(line+" , "+";".join(t1_P)+";".join(t2_P)+"\n")
            
        if flag_positive==1:
                flag_positive=0
                if tag=="0":
                    fp=fp+1
                    f_score.write(data_w)
                elif tag=="1":
                    tp=tp+1
                elif tag=="2":
                    tp_2+=1
        
        f_w.flush()
    
        f_w_not_handled_1.flush()
        
        f_w_rule_t2_in_t1.flush()
        f_w_rule_t1_in_t2.flush()
        
        f_w_whole_t2_in_t1.flush()
        f_w_whole_t1_in_t2.flush()
        
        f_w_affix_t2_in_t1.flush()
        f_w_affix_t1_in_t2.flush()    

    try:
        print "tp_2",tp_2
        print "tp",tp
        print "fp",fp
        print "all_p",all_p
        print "all_p_2",all_p_2
        precision=float(tp)/float(tp+fp)
        precision_naacl = float(tp+tp_2)/float(tp+tp_2+fp)
        print "precision",float(tp)/float(tp+fp)
        print "precision naacl",float(tp+tp_2)/float(tp+tp_2+fp)
        recall=float(tp)/float(all_p)
        print "recall",float(tp)/float(all_p)
        recall_naacl  =float(tp+tp_2)/float(all_p+all_p_2)
        print "recall naacl",float(tp+tp_2)/float(all_p+all_p_2)
        F_score=2*precision*recall/(precision+recall)
        print "F-score",F_score
        F_score_naacl=2*precision_naacl*recall_naacl/(precision_naacl+recall_naacl)
        print "F-score naacl",F_score_naacl
        f_score.write("tp "+str(tp)+"\n")
        f_score.write("fp "+str(fp)+"\n")
        f_score.write("all_p "+str(all_p)+"\n")
        f_score.write("precision "+str(precision)+"\n")
        f_score.write("recall "+str(recall)+"\n")
        f_score.write("F_score "+str(F_score)+"\n")
    except:
        print "error"
    
    f_w_not_handled_1.close()
    f_w_whole_t2_in_t1.close()
    f_w_whole_t1_in_t2.close()
    f_w_affix_t2_in_t1.close()
    f_w_affix_t1_in_t2.close()
    f_score.close()
f_w.close()
f_log.close()
