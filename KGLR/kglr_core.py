'''
This file contains all operators (functions)
'''
import re,sys,os,nltk,itertools
import MySQLdb as mdb
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.tag.stanford import StanfordPOSTagger
#from kglr_settings import *
import kglr_settings
from kglr_utility import *

lemmatizer=get_lemmatizer()

'''
Deverbal Nouns
This function checks if the given noun is deverbal form of given verb word. Example, director (n) - directs (v)
'''
def nounify_guided(verb_word,noun_word):
    set_of_related_nouns = set()
    if wn.morphy(verb_word, wn.VERB):
        for lemma in wn.lemmas(wn.morphy(verb_word, wn.VERB), pos="v"):
            for related_form in lemma.derivationally_related_forms():
                for synset in wn.synsets(related_form.name(), pos=wn.NOUN):
                    if wn.synset('person.n.01') in synset.closure(lambda s:s.hypernyms()):
                        if synset.lemmas()[0].name()==noun_word:
                            return 1
    return 0

'''(E)
Preposition Synonyms
'''
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


'''
Wordnet Hypernyms
flag_hypernym (1 if RHS is a hypernym of LHS),flag_hyponym (currently not used: 0 all time),flag_common_hypernym (LHS and RHS have a common hypernym)
'''
def check_relation(t1_P_clip,t1_L_clip,t2_L_clip,t2_P_clip,level,flag_positive,prev_dir,f_th,data_w,direction,flag_write):
    pos_t1=""
    if len(t1_P_clip)==1:
        for morphy_tag_key in kglr_settings.morphy_tag:
            if re.search("^"+morphy_tag_key,t1_P_clip[0].split("', '")[1].strip("('").strip("')")):
                pos_t1=kglr_settings.morphy_tag[morphy_tag_key]
                break
    
    pos_t2=""
    if len(t2_P_clip)==1:
        for morphy_tag_key in kglr_settings.morphy_tag:
            if re.search("^"+morphy_tag_key,t2_P_clip[0].split("', '")[1].strip("('").strip("')")):
                pos_t2=kglr_settings.morphy_tag[morphy_tag_key]
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
            if kglr_settings.flag_level2_hypernyms:
                ele_level2_hyp=ele.hypernyms()
                for ele_level2_hyp_ele in ele_level2_hyp:
                    t1_syn_ele_hyper_lemma+=ele_level2_hyp_ele.lemmas()
        t1_syn_ele_hyper_lemma_str=[str(ele) for ele in t1_syn_ele_hyper_lemma]
        if set(t1_syn_ele_hyper_lemma_str).intersection(set(t2_syn_lemma_str)):
            flag_hypernym=1
            break
    return flag_hypernym,flag_hyponym,flag_common_hypernym

'''
Thesaurus Synonym
'''
def check_thesaurus(t1_P_clip,t1_L_clip,t2_L_clip,t2_P_clip,level,flag_positive,prev_dir,f_th,data_w,direction,flag_write):
    flag_found=0
    kglr_settings.search_str=""
    if kglr_settings.rule_verbverb and not(direction=="rev"):
        t1_P_clip,t1_L_clip,flag_found=check_verbverb_construction(t1_P_clip,t1_L_clip,t2_L_clip)
        if not flag_positive and t1_L_clip and t2_L_clip: 
            t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L_clip, t2_L_clip,t1_P_clip,t2_L_clip)
            if not(t1_affix) and not(t2_affix):
                flag_positive=1
                kglr_settings.f_w_rule_t2_in_t1.write(data_w)
                        
    if type(t1_L_clip) is list:
        t1_L_clip=" ".join(t1_L_clip)
                
    temp_t1_L_clip=(t1_L_clip.split())
    lemm_pos=""
    search_str_1=""
    search_str_2=""
                
    if len(temp_t1_L_clip)==1:
        if kglr_settings.flag_pos_match and (re.search("\(",t1_P_clip[0])):
            if (re.search("VB",t1_P_clip[0].split("', '")[1].strip("('").strip("')"))):
                kglr_settings.search_str="verb"
                lemm_pos=wn.VERB
            elif (re.search("NN",t1_P_clip[0].split("', '")[1].strip("('").strip("')"))):
                    kglr_settings.search_str="noun"
                    lemm_pos=wn.NOUN
            elif (re.search("JJ",t1_P_clip[0].split("', '")[1].strip("('").strip("')"))):
                    kglr_settings.search_str="adj"
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
        if kglr_settings.flag_pos_match:
            if (re.search("VB",t2_P_clip[0].split("', '")[1].strip("('").strip("')"))):
                lemm_pos=wn.VERB
                search_str_2="verb"
            elif (re.search("NN",t2_P_clip[0].split("', '")[1].strip("('").strip("')"))):
                lemm_pos=wn.NOUN
                search_str_2="noun"#"verb"
            elif (re.search("JJ",t2_P_clip[0].split("', '")[1].strip("('").strip("')"))):
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
    search_str_1=kglr_settings.search_str
    if (re.search("verb",search_str_1) and re.search("adj",search_str_2)) or (re.search("verb",search_str_2) and re.search("adj",search_str_1)):
        kglr_settings.search_str="(adj|verb)"
        search_str_2=search_str_1=kglr_settings.search_str
    tmp_list=get_synonyms(temp_t1_L_clip)
    kglr_settings.search_str=""

    if len(tmp_list)>0:
        if len(t2_L_clip.split())>0 and len(t1_L_clip.split())>0:                
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
            kglr_settings.count_t2int1_prev=count_t2int1                    
            if del1=="True":
                prev_dir=1
                if flag_positive==0 and not kglr_settings.flag_bidir_syn:
                    flag_positive=1
                    f_th.write(data_w)
                    f_th.flush()
                if flag_positive==0 and flag_found and kglr_settings.flag_bidir_syn and kglr_settings.rule_verbverb:
                    del1,flag_positive,prev_dir=check_thesaurus(t2_P_clip,t2_L_clip,t1_L_clip,t1_P_clip,level,flag_positive,prev_dir,f_th,data_w,"rev",flag_write)
            return(del1,flag_positive,prev_dir)       
        elif level ==2:
            if del1=="True":
                if kglr_settings.flag_bidir_syn and prev_dir and not flag_positive:
                    kglr_settings.search_str=search_str_1
                    count_t1=get_count(t1_L_clip)#+0.00000001
                    kglr_settings.search_str=search_str_2
                    count_t2=get_count(t2_L_clip)#+0.00000001                  
                    kglr_settings.search_str=""
                    
                    if kglr_settings.flag_affix_call==0:
                        f_th.write(data_w)
                        flag_positive=1
                    else:
                        if count_t1 and count_t2:
                            count_data_val=(float(kglr_settings.count_t2int1_prev)+float(count_t2int1))/float(count_t1*count_t2)#count_data_val=(float(kglr_settings.count_t2int1_prev)+float(count_t2int1))/float(count_t1+count_t2)#/float(count_t1*count_t2)
                        else:
                            count_data_val=0
                        if count_data_val >=0.003:#count_data_val>=0.1:#count_data_val >=0.003:
                            f_th.write(data_w)
                            flag_positive=1
                        else:
                            flag_positive=0
                    f_th.flush()
                if flag_positive==0 and flag_found and kglr_settings.flag_bidir_syn and kglr_settings.rule_verbverb:
                    prev_dir=1
                    del1,flag_positive,prev_dir=check_thesaurus(t2_P_clip,t2_L_clip,t1_L_clip,t1_P_clip,level,flag_positive,prev_dir,f_th,data_w,"rev",flag_write)
            prev_dir=0
            kglr_settings.count_t2int1_prev=0                                
            return(del1,flag_positive,prev_dir)         
    else:
        prev_dir=0
        del1="False"
        if len(kglr_settings.check_thesaurus_100)>=100:
            kglr_settings.check_thesaurus_100.popitem()
        kglr_settings.check_thesaurus_100[str(t1_P_clip)+":"+str(t1_L_clip)+":"+str(t2_L_clip)+":"+str(level)+":"+str(flag_positive)+":"+str(prev_dir)]=del1,flag_positive,prev_dir
            
        return(del1,flag_positive,prev_dir)

'''(E)
Light Verbs and Serial Verbs
Remove light/serial verb if exists
'''
def check_lightverb_construction(t1_P_lightverbDT,t1_L_lightverbDT):
    flag_found=0
    if len(t1_P_lightverbDT)>2 and len(t1_L_lightverbDT)>2:
        if (t1_L_lightverbDT[0] in kglr_settings.light_verb_old) and (re.search("^(a|an)",t1_P_lightverbDT[1].split("', '")[0].strip("('").strip("')"))) and (re.search("^(NN|VB)",t1_P_lightverbDT[2].split("', '")[1].strip("('").strip("')"))):
            t1_P_lightverbDT=t1_P_lightverbDT[2:]
            t1_L_lightverbDT=t1_L_lightverbDT[2:]
            flag_found=1
    return t1_P_lightverbDT,t1_L_lightverbDT,flag_found

def check_verbverb_construction(t1_P_lightverbDT,t1_L_lightverbDT,t2_L_lightverbDT):
    flag_found=0
    if not type(t1_L_lightverbDT) is list:
        t1_L_lightverbDT=t1_L_lightverbDT.split()
    if not type(t2_L_lightverbDT) is list:
        t2_L_lightverbDT=t2_L_lightverbDT.split()
    if len(t1_P_lightverbDT)>1 and len(t1_L_lightverbDT)>1:
        if (t1_L_lightverbDT[0] in kglr_settings.all_verb_verb) and ((re.search("^(NN|VB)",t1_P_lightverbDT[1].split("', '")[1].strip("('").strip("')")))):
            t1_P_lightverbDT=t1_P_lightverbDT[1:]
            t1_L_lightverbDT=t1_L_lightverbDT[1:]
            flag_found=1
    if len(t1_P_lightverbDT)>2 and len(t1_L_lightverbDT)>2:
        if (t1_L_lightverbDT[0] in kglr_settings.all_verb_verb) and (re.search("^(DT)",t1_P_lightverbDT[1].split("', '")[1].strip("('").strip("')")) and re.search("^(NN|VB)",t1_P_lightverbDT[2].split("', '")[1].strip("('").strip("')"))):
            t1_P_lightverbDT=t1_P_lightverbDT[2:]
            t1_L_lightverbDT=t1_L_lightverbDT[2:]
            flag_found=1    
    return t1_P_lightverbDT,t1_L_lightverbDT,flag_found

'''(E)
Dropping Modifiers (Superlatives)
'''
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

'''(E)
Dropping Modifiers (Adjective) heper function:
extracts JJ from rule (if they precede NN & JJ)
returns extracted JJ and remainder rel
'''
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

'''(E)
Gerund-Infinitive Equivalence
converts <starts to walk> --> <start walking>
if starts is in L_gerund_infi_to
'''
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

'''(E)
Dropping DT
Drops DT (a/an/the) from the left and right boundry of the phrase
'''
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

'''(E)
Dropping DT
Drops DT (a/an/the/all/...etc) from the left and right boundry of the phrase
'''
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

'''
Active-Passive
'''
def rule_active_passive_be_fn(direction,t1_L_rule_ap,t2_L_rule_ap,t1_P_rule_ap,t2_P_rule_ap,del1,flag_positive,prev_dir,data_w):
    if t1_L_rule_ap and t2_L_rule_ap:
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
            #start: termination condition check (type2)
            if t2_L_rule_ap and t1_L_rule_ap:
                derivation_t2=get_derivation(" ".join(t2_L_rule_ap),t1_P_rule_ap)
                derivation_t1=get_derivation(" ".join(t1_L_rule_ap),t1_P_rule_ap)
                for word_derivation in derivation_t2:
                    if direction=="fwd":
                        del1,flag_positive,prev_dir=check_thesaurus(t1_P_rule_ap," ".join(t1_L_rule_ap), word_derivation,t2_P_rule_ap, 1, flag_positive, prev_dir,kglr_settings.f_w_rule_t1_in_t2,data_w,"fwd",1)
                        del1,flag_positive,prev_dir=check_thesaurus(t2_P_rule_ap,word_derivation," ".join(t1_L_rule_ap),t1_P_rule_ap, 2, flag_positive, prev_dir,kglr_settings.f_w_rule_t2_in_t1,data_w,"rev",1)
                    else:
                        del1,flag_positive,prev_dir=check_thesaurus(t1_P_rule_ap," ".join(t1_L_rule_ap), word_derivation,t2_P_rule_ap, 1, flag_positive, prev_dir,kglr_settings.f_w_rule_t2_in_t1,data_w,"rev",1)
                        del1,flag_positive,prev_dir=check_thesaurus(t2_P_rule_ap,word_derivation," ".join(t1_L_rule_ap),t1_P_rule_ap, 2, flag_positive, prev_dir,kglr_settings.f_w_rule_t1_in_t2,data_w,"fwd",1)
                if not flag_positive:
                    for word_derivation in derivation_t1:
                        if direction=="fwd":
                            del1,flag_positive,prev_dir=check_thesaurus(t2_P_rule_ap," ".join(t2_L_rule_ap), word_derivation,t1_P_rule_ap, 1, flag_positive, prev_dir,kglr_settings.f_w_rule_t2_in_t1,data_w,"rev",1)
                            del1,flag_positive,prev_dir=check_thesaurus(t1_P_rule_ap,word_derivation, " ".join(t2_L_rule_ap),t2_P_rule_ap, 2, flag_positive, prev_dir,kglr_settings.f_w_rule_t1_in_t2,data_w,"fwd",1)
                        else:
                            del1,flag_positive,prev_dir=check_thesaurus(t2_P_rule_ap," ".join(t2_L_rule_ap), word_derivation,t1_P_rule_ap, 1, flag_positive, prev_dir,kglr_settings.f_w_rule_t1_in_t2,data_w,"fwd",1)
                            del1,flag_positive,prev_dir=check_thesaurus(t1_P_rule_ap,word_derivation, " ".join(t2_L_rule_ap),t2_P_rule_ap, 2, flag_positive, prev_dir,kglr_settings.f_w_rule_t2_in_t1,data_w,"rev",1)           
            #end: termination condition check (type2)
    return t1_L_rule_ap,t2_L_rule_ap,t1_P_rule_ap,t2_P_rule_ap,del1,flag_positive,prev_dir
