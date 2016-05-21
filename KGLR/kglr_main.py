'''

'''
import re,sys,os,nltk,itertools
import MySQLdb as mdb
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.tag.stanford import StanfordPOSTagger

# Global_data={}
# ###Set paths###
# java_path = "C:\\Program Files\\Java\\jdk1.8.0_60\\bin\\java.exe"
# postagger_model_path='E:\\nltk_data\\stanford\\stanford-postagger.jar'
# distsimtagger_model_path='E:\\nltk_data\\stanford\\model\\english-bidirectional-distsim.tagger'
# path="E:\\EclipseIndigo\\workspace\\Inference\\"#project_path
# rule_in_file="in_files\\ablation.txt"#input file
# rule_out_file=""#output file
# #Data files#
# redundant_prep_file="data\\ch9out2-r-0000all_norm_IN_null.txt"
# prep_syn_file="data\\prep_pair_all_freq.txt"
# java_options_mem='-mx2G'
# #thesaurus db#
# Global_data['thesaurus_db'] = 'thesaurus'
# Global_data['password_db']=''
# ###Set paths###
# # ###Set paths:Ubuntu###
# # java_path = "/usr/lib/jvm/java-8-oracle/bin/java"
# # postagger_model_path='/home/prachi/nltk_data/stanford/stanford-postagger.jar'
# # distsimtagger_model_path='/home/prachi/nltk_data/stanford/model/english-bidirectional-distsim.tagger'
# # java_options_mem='-mx20G'
# # path="/home/prachi/Documents/project/code/ver_naacl/Inference/"
# # rule_in_file="in_files/ablation.txt"
# # rule_out_file=""
# # redundant_prep_file="data/ch9out2-r-0000all_norm_IN_null.txt"
# # prep_syn_file="data/prep_pair_all_freq.txt"
# # thesaurus_db = 'thesaurus'
# # password_db='nlp_prachi'
# # ###Set paths:Ubuntu###
# 
# 
# ###Feature flags###
# #Thesaurus Synonyms
# Global_data['flag_whole']=1
# Global_data['flag_antonyms']=1
# Global_data['flag_syn_of_syn']=1
# #Negating rules
# Global_data['flag_not']=1
# #Wordnet Hypernyms
# Global_data['flag_wordnet_hypernym_hyponym']=1
# Global_data['flag_level2_hypernyms']=1
# #Dropping Modifiers
# Global_data['rule_superlative']=1
# Global_data['rule_JJ_NN_mod']=1
# #Gerund-Infinitive Equivalence
# Global_data['flag_gerund_infi_to']=1
# #Deverbal Nouns
# Global_data['flag_noun_verb_wn']=1
# #Light Verbs and Serial Verbs
# Global_data['rule_lightverb_dt']=1
# Global_data['rule_verbverb']=1
# #Preposition Synonyms
# Global_data['rule_equal_preposition']=1
# Global_data['rule_equal_preposition_data_freq']=1
# #Be-words & Determiners
# Global_data['flag_be_trail']=1
# #flag_dt=1#dt rule
# Global_data['flag_deep_dt']=1
# Global_data['kglr_settings.f_logdt']=1#interim removal of dt
# Global_data['rule_be']=1
# #Active-Passive
# Global_data['rule_active_passive_be']=1
# #Redundant Preposition
# Global_data['rule_preposition_null']=1#X;learn about;Y --> X;learn;Y
# #Extra#be - preposition#("be sick of@R@;tire of@R@"("be sick of@R@;tire of@R@")
# Global_data['rule_be_prep']=1
# 
# ####Utility Flags###
# #del
# Global_data['flag_rules']=1
# Global_data['flag_stanford_pos']=1
# 
# Global_data['flag_affix']=1
# #utility
# Global_data['count_t2int1_prev']=0
# Global_data['flag_affix_call']=0
# Global_data['flag_rev']=1#to ensure: X t1 Y -> X t2 Y
# Global_data['flag_bidir_syn']=1#flag to check t1 in t2'syn and t2 in t1's syn
# Global_data['flag_pos_match']=1#ensure matching of same pos words
# Global_data['flag_derivation']=1#extract derivations of given word
# 
# #extra
# Global_data['rule_have']=1
# Global_data['rule_have_JJ']=1
# 
# ####Init values###
# Global_data['tp_2']=0
# Global_data['tp']=0
# Global_data['kglr_settings.flag_']=0
# Global_data['all_p']=0
# Global_data['all_p_2'] = 0
# Global_data['flag_positive']=1
# 
# Global_data['check_thesaurus_100']={}#for speedup
# Global_data['check_db_100']={}#for speedup
# Global_data['count_t2int1_prev']=0
# Global_data['flag_affix_call']=0
# 
# #aux_verb=["be"]
# Global_data['light_verb_old']=["take", "have", "give", "do", "make"]
# Global_data['verbs_verb']=["has","have","be","is","were","are","was","had","being","began","am","following","having","do","does",
#             "did","started","been","became","left","help","helped","get","keep","think","got","gets","include",
#             "suggest","used","see","consider","means","try","start","included","lets","say","continued",
#             "go","includes","becomes","begins","keeps","begin","starts","said"]#,"stop"
# Global_data['verb_verb_norm']=["begin","start","continue","say"]
# Global_data['all_verb_verb']=Global_data['verb_verb_norm']+Global_data['verbs_verb']#used for checking serial verb constructions
# Global_data['L_gerund_infi_to']=["attempt","begin","bother","cease","continue","deserve","neglect","omit","permit","start","fear","intend","recommend","advice","allow","permit","encourage","forbid","choose"]
# Global_data['L_gerund_infi_to_t1']=["like","love","prefer"]
import kglr_settings
#from kglr_settings import *
kglr_settings.init()
#init()
#from kglr_settings import *

import kglr_core,kglr_utility
from kglr_core import *
from kglr_utility import *

#lemmatizer=get_lemmatizer()#kglr_settings.java_path,kglr_settings.distsimtagger_model_path,kglr_settings.postagger_model_path,kglr_settings.java_options_mem)

if __name__ == '__main__':
    lines=read_file(kglr_settings.path+kglr_settings.rule_in_file)
    
    ###Redundant preposition data###
    prep_null=get_null_preposition_data()
    
    ###Preposition Synonyms data###
    freq_prep=get_syn_preposition_data()
        
    ###POS TAG of rule words###
    stanford_pos_dict=pos_tag_rules(lines)
    
    for line in lines:
        flag_positive=0
        kglr_settings.flag_affix_call=0
        #preprocess start
        line,L_line,line_w,flag_rev_gate,tag,t1,t1_L,t1_P,t2,t2_L,t2_P = prepare_node(line,stanford_pos_dict)
        
        #pre-processed data cleaning            
        t1_L,t1_P,t2_L,t2_P,data_w=clean_data(t1_L, t1_P, t2_L, t2_P, line_w,line)           
        
        #relation phrase cannot end with "be"
        if kglr_settings.flag_be_trail:
            if re.search(" be$",t1) or re.search(" be$",t2):
                continue
        
        #Drop Determiners on the boundry and check if LHS==RHS
        if kglr_settings.rules_dt:
            t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L, t2_L,t1_P,t2_P)    
            if len(t1_affix)>0:
                t1_P_affix,t1_affix=strip_DT(t1_P_affix,t1_affix)
            if len(t2_affix):
                t2_P_affix,t2_affix=strip_DT(t2_P_affix,t2_affix)
            if not (t1_affix) and not(t2_affix):
                flag_positive=1
                kglr_settings.f_w_affix_t1_in_t2.write(line_w+"\n")
        
        #Branch 1
        #Thesaurus synonyms : check if whole rel1 (LHS) and whole rel2 (RHS) are synonymous:: Overlap of syn(LHS) and syn(RHS) is high
        if kglr_settings.flag_whole and flag_rev_gate:
            prev_dir=0
            flag_derivation=0
            del1,flag_positive,prev_dir=check_thesaurus(t1_P,t1, t2,t2_P, 1, flag_positive, prev_dir,kglr_settings.f_w_whole_t2_in_t1,data_w,"fwd",1)
            kglr_settings.f_w.write(del1+" , ")
            del1,flag_positive,prev_dir=check_thesaurus(t2_P,t2, t1,t1_P, 2, flag_positive, prev_dir,kglr_settings.f_w_whole_t1_in_t2,data_w,"rev",1)
            kglr_settings.f_w.write(del1+" , ")
            if not flag_positive and not (re.search(t1+"$",t2) or re.search(t2+"$",t1)) and not re.search("^be", t1) and not re.search("^be", t2):
                tmp_list_t1=get_synonyms([t1])
                if len(tmp_list_t1)>0 and len(t1.split()) > 1:
                    kglr_settings.f_log.write("t1 -- >"+line+"\n")
                    if tag=="1":
                        kglr_settings.f_w_not_handled_1.write(line+" , "+";".join(t1_P)+";".join(t2_P)+"\n")
                    continue
                tmp_list_t2=get_synonyms([t2])
                if len(tmp_list_t2)>0 and len(t2.split()) > 1:
                    kglr_settings.f_log.write("t2 -- >"+line+"\n")
                    if tag=="1":
                        kglr_settings.f_w_not_handled_1.write(line+" , "+";".join(t1_P)+";".join(t2_P)+"\n")
                    continue
            flag_derivation=1
            
        #Branch 2
        #Handling NOT: if word(LHS) - word(RHS) = "not": Drop the rule
        if kglr_settings.flag_not and kglr_settings.flag_affix and not flag_positive:
            prev_dir=0
            kglr_settings.flag_affix_call=1
            t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L, t2_L,t1_P,t2_P)
            if kglr_settings.rules_dt:
                if len(t1_affix)>0:
                    t1_P_affix,t1_affix=strip_DT(t1_P_affix,t1_affix)
                if len(t2_affix):
                    t2_P_affix,t2_affix=strip_DT(t2_P_affix,t2_affix)
            
            if (not t1_affix and t2_affix=="not") or(not t2_affix and t1_affix=="not"):
                kglr_settings.flag_affix_call=0
                continue 
            kglr_settings.flag_affix_call=0

        #Branch 3
        t1_L_rule=t1_L
        t2_L_rule=t2_L
        t1_P_rule=t1_P
        t2_P_rule=t2_P
        rule_have_fire=0
        
        ###Branch 3.1: Gerund-Infinitive Equivalence Rule###
        if kglr_settings.flag_gerund_infi_to:
            t1_P_rule,t1_L_rule,check_gerund_infi_found=check_gerund_infi(t1_P_rule,t1_L_rule,kglr_settings.L_gerund_infi_to)
            t2_P_rule,t2_L_rule,check_gerund_infi_found=check_gerund_infi(t2_P_rule,t2_L_rule,kglr_settings.L_gerund_infi_to)
            t1_P_rule,t1_L_rule,check_gerund_infi_found=check_gerund_infi(t1_P_rule,t1_L_rule,kglr_settings.L_gerund_infi_to_t1)
            if not flag_positive: 
                t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L_rule, t2_L_rule,t1_P_rule,t2_P_rule)
                if not(t1_affix) and not(t2_affix):
                    flag_positive=1
                    kglr_settings.f_w_rule_t2_in_t1.write(data_w)
        
        ###Branch 3.2: Drop Modifiers (Superlatives)###        
        if kglr_settings.rule_superlative:#more most
            t1_P_rule,t1_L_rule=check_superlative(t1_P_rule,t1_L_rule)
            t2_P_rule,t2_L_rule=check_superlative(t2_P_rule,t2_L_rule)
            
        ###Drop Modifiers (Adjective): Adjectives are dropped only if they are seen in t1 only: If syn adj are seen on LHS and RHS, they are dropped: We look at JJ which precede JJ|NN### 
        if kglr_settings.rule_JJ_NN_mod:
            t1_P_rule_back,t1_L_rule_back,t1_JJ_L_rem,t1_JJ_P_rem=check_JJNN(t1_P_rule,t1_L_rule)
            t2_P_rule_back,t2_L_rule_back,t2_JJ_L_rem,t2_JJ_P_rem=check_JJNN(t2_P_rule,t2_L_rule)
            if t1_JJ_L_rem and t2_JJ_L_rem:#Branch 3.3a:Go in if both t1 and t2 have JJ which precedes NN or JJ::Checks if every JJ of t1 is synonymous to corresponding JJ of t2 
                JJ_flag_positive=0
                JJ_flag_positive_back=1
                for temp_i in range(0,len(t1_JJ_L_rem)):#last JJ match
                    del1,JJ_flag_positive,prev_dir=check_thesaurus([t1_JJ_P_rem[temp_i]],[t1_JJ_L_rem[temp_i]],[t2_JJ_L_rem[temp_i]],[t2_JJ_P_rem[temp_i]],1,0,prev_dir,kglr_settings.f_w_rule_t1_in_t2,"","fwd",0)
                    del1,JJ_flag_positive,prev_dir=check_thesaurus([t2_JJ_P_rem[temp_i]],[t2_JJ_L_rem[temp_i]],[t1_JJ_L_rem[temp_i]],[t1_JJ_P_rem[temp_i]],2,JJ_flag_positive,prev_dir,kglr_settings.f_w_rule_t1_in_t2,"","rev",0)
                    JJ_flag_positive_back=JJ_flag_positive_back and JJ_flag_positive
                JJ_flag_positive=JJ_flag_positive_back
                    
                if JJ_flag_positive:
                    kglr_settings.f_log.write(line_w)
                    t2_P_rule=t2_P_rule_back
                    t2_L_rule=t2_L_rule_back
                    t1_P_rule=t1_P_rule_back
                    t1_L_rule=t1_L_rule_back
                    #start: termination condition check
                    prev_dir=0
                    t1_affix_rule_back,t2_affix_rule_back,t1_P_affix_rule_back,t2_P_affix_rule_back=remove_common_affix(t1_L_rule_back, t2_L_rule_back,t1_P_rule_back,t2_P_rule_back)
                    
                    if kglr_settings.rules_dt:
                        if len(t1_affix_rule_back)>0:
                            t1_P_affix_rule_back,t1_affix_rule_back=strip_DT(t1_P_affix_rule_back,t1_affix_rule_back)
                        if len(t2_affix_rule_back):
                            t2_P_affix_rule_back,t2_affix_rule_back=strip_DT(t2_P_affix_rule_back,t2_affix_rule_back)
                    
                    t1_affix_rule_back=" ".join(t1_affix_rule_back)
                    t2_affix_rule_back=" ".join(t2_affix_rule_back)
                    del1,flag_positive,prev_dir=check_thesaurus(t1_P_affix_rule_back,t1_affix_rule_back, t2_affix_rule_back,t2_P_affix_rule_back, 1, flag_positive, prev_dir,kglr_settings.f_w_affix_t2_in_t1,data_w,"fwd",1)
                    kglr_settings.f_w.write(del1+" , ")
                    del1,flag_positive,prev_dir=check_thesaurus(t2_P_affix_rule_back,t2_affix_rule_back, t1_affix_rule_back,t1_P_affix_rule_back, 2, flag_positive, prev_dir,kglr_settings.f_w_affix_t1_in_t2,data_w,"rev",1)
                    kglr_settings.f_w.write(del1+" , ")
                    #end: termination condition check 
            elif t1_L_rule_back:#Branch 3.3b:Drop JJ if there is no corresonding JJ in t2::Improvement - accept this for only JJ
                t2_P_rule=t2_P_rule_back
                t2_L_rule=t2_L_rule_back
                t1_P_rule=t1_P_rule_back
                t1_L_rule=t1_L_rule_back
                
            if not flag_positive: 
                #start: termination condition check
                t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L_rule, t2_L_rule,t1_P_rule,t2_P_rule)
                if not(t1_affix) and not(t2_affix):
                    flag_positive=1
                    kglr_settings.f_w_rule_t2_in_t1.write(data_w)
                #end: termination condition check
                
        #Branch 3.4: Drop Light verbs and Serial verbs only from t1          
        if kglr_settings.rule_lightverb_dt:
            t1_P_rule_lightverbDT=t1_P_rule
            t2_P_rule_lightverbDT=t2_P_rule
            t1_L_rule_lightverbDT=t1_L_rule
            t2_L_rule_lightverbDT=t2_L_rule
            flag_found=0
            
            t1_P_rule,t1_L_rule,flag_found=check_lightverb_construction(t1_P_rule_lightverbDT,t1_L_rule_lightverbDT)
            if not flag_positive:
                #start: termination condition check 
                t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L_rule, t2_L_rule,t1_P_rule,t2_P_rule)
                if not(t1_affix) and not(t2_affix):
                    flag_positive=1
                    kglr_settings.f_w_rule_t2_in_t1.write(data_w)
                #end: termination condition check
        
        #Branch 3.5:        
        if kglr_settings.rule_have_JJ:
            #have a huge impact on --> have impact on #applied on t1 as well as t2 
            t1_P_rule_haveJJ=t1_P_rule;t2_P_rule_haveJJ=t2_P_rule;t1_L_rule_haveJJ=t1_L_rule;t2_L_rule_haveJJ=t2_L_rule
            t1_P_rule_haveJJ_strip=[];t2_P_rule_haveJJ_strip=[];t1_L_rule_haveJJ_strip=[];t2_L_rule_haveJJ_strip=[]
            
            #check for have JJ pattern on t1
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

            #check for have JJ pattern on t2
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
            have_JJ_update=0
        
        #Branch 3.6    
        if kglr_settings.rule_have:
            #"1";have responsibility for;be accountable for
            if t1_L_rule[0]=="have":
                pos_ele_2=t2_P_rule[0].split("', '")[1].strip("('").strip("')")
                if t2_L_rule[0]=="be":
                    t1_L_rule=t1_L_rule[1:]
                    t2_L_rule=t2_L_rule[1:]
                    t1_P_rule=t1_P_rule[1:]
                    t2_P_rule=t2_P_rule[1:]
                    #start: termination condition check
                    prev_dir=0
                    del1,flag_positive,prev_dir=check_thesaurus(t1_P_rule,t1_L_rule, t2_L_rule,t2_P_rule, 1, flag_positive, prev_dir,kglr_settings.f_w_rule_t2_in_t1,data_w,"fwd",1)
                    del1,flag_positive,prev_dir=check_thesaurus(t2_P_rule,t2_L_rule,t2_P_rule, t1_L_rule, 2, flag_positive, prev_dir,kglr_settings.f_w_rule_t1_in_t2,data_w,"rev",1)
                    rule_have_fire=1
                    if not flag_positive: 
                        t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L_rule, t2_L_rule,t1_P_rule,t2_P_rule)
                        if not(t1_affix) and not(t2_affix):
                            flag_positive=1
                            kglr_settings.f_w_rule_t2_in_t1.write(data_w)
                    #end: termination condition check
        
        #start: termination condition check            
        if kglr_settings.rules_dt:
            t1_P_rule,t1_L_rule=strip_DT(t1_P_rule,t1_L_rule)
            t2_P_rule,t2_L_rule=strip_DT(t2_P_rule,t2_L_rule)
            if not flag_positive: 
                t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L_rule, t2_L_rule,t1_P_rule,t2_P_rule)
                if not(t1_affix) and not(t2_affix):
                    flag_positive=1
                    kglr_settings.f_w_rule_t2_in_t1.write(data_w)
        #end: termination condition check
        
        #Branch 3.7: 'be' dropped from start of t1 & t2    
        if kglr_settings.rule_be:
            if t1_L_rule[0]=="be" and t2_L_rule[0]=="be":
                #print "\n",t1_L_rule,t2_L_rule
                t1_L_rule=t1_L_rule[1:]
                t1_P_rule=t1_P_rule[1:]   
                #print t1_L_rule,t2_L_rule
                t2_L_rule=t2_L_rule[1:]
                t2_P_rule=t2_P_rule[1:]
                if not flag_positive: 
                    #start: termination condition check
                    t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L_rule, t2_L_rule,t1_P_rule,t2_P_rule)
                    if not(t1_affix) and not(t2_affix):
                        flag_positive=1
                        kglr_settings.f_w_rule_t2_in_t1.write(data_w)
                    #end: termination condition check
        
        #start: termination condition check            
        if kglr_settings.rules_dt:
            t1_P_rule,t1_L_rule=strip_DT(t1_P_rule,t1_L_rule)
            t2_P_rule,t2_L_rule=strip_DT(t2_P_rule,t2_L_rule)
            if not flag_positive: 
                t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L_rule, t2_L_rule,t1_P_rule,t2_P_rule)
                if not(t1_affix) and not(t2_affix):
                    flag_positive=1
                    kglr_settings.f_w_rule_t2_in_t1.write(data_w)
        #end: termination condition check    
        
        #Branch 3.8: If t1|t2 starts with be and t2|t1 doesn't + both end with same preposition then drop be from t1|t2 and trailing preposition from both t1 & t2.             
        if kglr_settings.rule_be_prep:
            #if re.search("be sick of@R@;tire of@R@",line):
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
                    
                    #start: termination condition check (type2)           
                    if t2_L_rule_ap and t1_L_rule_ap:
                        derivation_t2=get_derivation(" ".join(t2_L_rule_ap),t2_P_rule_ap)
                        derivation_t2.append(" ".join(t2_L_rule_ap))               
                        derivation_t1=get_derivation(" ".join(t1_L_rule_ap),t1_P_rule_ap)
                        derivation_t1.append(" ".join(t1_L_rule_ap)) 
                        for word_derivation in derivation_t2:
                            del1,flag_positive,prev_dir=check_thesaurus(t1_P_rule_ap," ".join(t1_L_rule_ap), word_derivation, t2_P_rule_ap,1, flag_positive, prev_dir,kglr_settings.f_w_rule_t2_in_t1,data_w,"fwd",1)
                            del1,flag_positive,prev_dir=check_thesaurus(t2_P_rule_ap,word_derivation," ".join(t1_L_rule_ap),t1_P_rule_ap, 2, flag_positive, prev_dir,kglr_settings.f_w_rule_t1_in_t2,data_w,"rev",1)
                        if not flag_positive:
                            for word_derivation in derivation_t1:
                                del1,flag_positive,prev_dir=check_thesaurus(t2_P_rule_ap," ".join(t2_L_rule_ap), word_derivation,t1_P_rule_ap, 1, flag_positive, prev_dir,kglr_settings.f_w_rule_t2_in_t1,data_w,"rev",1)
                                del1,flag_positive,prev_dir=check_thesaurus(t1_P_rule_ap,word_derivation, " ".join(t2_L_rule_ap),t2_P_rule_ap, 2, flag_positive, prev_dir,kglr_settings.f_w_rule_t1_in_t2,data_w,"fwd",1)
                    if not flag_positive: 
                        t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L_rule, t2_L_rule,t1_P_rule,t2_P_rule)
                        if not(t1_affix) and not(t2_affix):
                            flag_positive=1
                            kglr_settings.f_w_rule_t2_in_t1.write(data_w)
                    #end: termination condition check            
                elif t2_L_rule_ap[0]=="be" and t1_L_rule_ap[0]!="be" and t2_L_rule_ap[len(t2_L_rule_ap)-1].strip()==t1_L_rule_ap[len(t1_L_rule_ap)-1].strip() and t2_P_rule_ap[len(t2_P_rule_ap)-1].split(', ')[1].strip("')")=="IN":#"by":
                    t2_L_rule_ap=t2_L_rule_ap[:-1]
                    t2_L_rule_ap=t2_L_rule_ap[1:]
                    t2_P_rule_ap=t2_P_rule_ap[:-1]
                    t2_P_rule_ap=t2_P_rule_ap[1:]
                    t1_L_rule_ap=t1_L_rule_ap[:-1]
                    t1_P_rule_ap=t1_P_rule_ap[:-1]                    
                    #start: termination condition check (type2)           
                    if t2_L_rule_ap and t1_L_rule_ap:
                        derivation_t2=get_derivation(" ".join(t2_L_rule_ap),t2_P_rule_ap)
                        derivation_t2.append(" ".join(t2_L_rule_ap))               
                        derivation_t1=get_derivation(" ".join(t1_L_rule_ap),t1_P_rule_ap)
                        derivation_t1.append(" ".join(t1_L_rule_ap))         
                        for word_derivation in derivation_t2:
                            del1,flag_positive,prev_dir=check_thesaurus(t1_P_rule_ap," ".join(t1_L_rule_ap), word_derivation,t2_P_rule_ap, 1, flag_positive, prev_dir,kglr_settings.f_w_rule_t2_in_t1,data_w,"fwd",1)
                            del1,flag_positive,prev_dir=check_thesaurus(t2_P_rule_ap,word_derivation, " ".join(t1_L_rule_ap),t1_P_rule_ap, 2, flag_positive, prev_dir,kglr_settings.f_w_rule_t1_in_t2,data_w,"rev",1)
                        if not flag_positive:
                            for word_derivation in derivation_t1:
                                del1,flag_positive,prev_dir=check_thesaurus(t2_P_rule_ap," ".join(t2_L_rule_ap), word_derivation,t1_P_rule_ap, 1, flag_positive, prev_dir,kglr_settings.f_w_rule_t2_in_t1,data_w,"rev",1)
                                del1,flag_positive,prev_dir=check_thesaurus(t1_P_rule_ap,word_derivation, " ".join(t2_L_rule_ap),t2_P_rule_ap, 2, flag_positive, prev_dir,kglr_settings.f_w_rule_t1_in_t2,data_w,"fwd",1)
                    if not flag_positive: 
                        t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L_rule, t2_L_rule,t1_P_rule,t2_P_rule)
                        if not(t1_affix) and not(t2_affix):
                            flag_positive=1
                            kglr_settings.f_w_rule_t2_in_t1.write(data_w)
                    #end: termination condition check (type2)           
                
        ###Branch 3.9: Preposition Synonym Rule: Drop trailing preposition if they are synonymous###                    
        if kglr_settings.rule_equal_preposition:
            flag_derivation=0
            equal_preposition_L=[("into","into"),("towards","towards"),("onto","onto"),("from","from"),("till","till"),("until","until"),("across","across"),("through","through"),("along","along"),("around","around"),("up","up"),("down","down"),("over","over"),("under","under"),("as","to be"),("in","at"),("in","by"),("in","to"),("on","on"),("for","for"),("of","of"),("to","to"),("about","about"),("with","with")]
            if kglr_settings.rule_equal_preposition_data_freq:
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
                #start: termination condition check (type2)
                del1,flag_positive,prev_dir=check_thesaurus(t1_P_rule," ".join(t1_L_rule), " ".join(t2_L_rule),t2_P_rule, 1, flag_positive, prev_dir,kglr_settings.f_w_rule_t2_in_t1,data_w,"fwd",1)
                del1,flag_positive,prev_dir=check_thesaurus(t2_P_rule," ".join(t2_L_rule), " ".join(t1_L_rule), t1_P_rule,2, flag_positive, prev_dir,kglr_settings.f_w_rule_t1_in_t2,data_w,"rev",1)
                if not flag_positive and prep_pair==("as","to be"):
                    derivation_t2=get_derivation(" ".join(t2_L_rule),t2_P_rule)
                    derivation_t2.append(" ".join(t2_L_rule))               
                    derivation_t1=get_derivation(" ".join(t1_L_rule),t1_P_rule)
                    derivation_t1.append(" ".join(t1_L_rule)) 
                    for word_derivation in derivation_t2:
                        del1,flag_positive,prev_dir=check_thesaurus(t1_P_rule," ".join(t1_L_rule), word_derivation,t2_P_rule, 1, flag_positive, prev_dir,kglr_settings.f_w_rule_t2_in_t1,data_w,"fwd",1)
                        del1,flag_positive,prev_dir=check_thesaurus(t2_P_rule,word_derivation," ".join(t1_L_rule),t1_P_rule, 2, flag_positive, prev_dir,kglr_settings.f_w_rule_t1_in_t2,data_w,"rev",1)
                    if not flag_positive:
                        for word_derivation in derivation_t1:
                            del1,flag_positive,prev_dir=check_thesaurus(t2_P_rule," ".join(t2_L_rule), word_derivation,t1_P_rule, 1, flag_positive, prev_dir,kglr_settings.f_w_rule_t2_in_t1,data_w,"rev",1)
                            del1,flag_positive,prev_dir=check_thesaurus(t1_P_rule,word_derivation, " ".join(t2_L_rule),t2_P_rule, 2, flag_positive, prev_dir,kglr_settings.f_w_rule_t1_in_t2,data_w,"fwd",1)
                if not flag_positive: 
                    t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L_rule, t2_L_rule,t1_P_rule,t2_P_rule)
                    if not(t1_affix) and not(t2_affix):
                        flag_positive=1
                        kglr_settings.f_w_rule_t2_in_t1.write(data_w) 
                #end: termination condition check (type2)           
            flag_derivation=1
            
        if rule_have_fire and (t1_L_rule or t2_L_rule):
            #start: termination condition check (type2)
            t1_a_rule,t2_a_rule,t1_ap_rule,t2_ap_rule=remove_common_affix(t1_L_rule,t2_L_rule,t1_P_rule,t2_P_rule)
            if t1_a_rule and t2_a_rule:
                derivation_t2=get_derivation(" ".join(t2_a_rule),t2_ap_rule)
                derivation_t2.append(" ".join(t2_a_rule))               
                derivation_t1=get_derivation(" ".join(t1_a_rule),t1_ap_rule)
                derivation_t1.append(" ".join(t1_a_rule)) 
                for word_derivation in derivation_t2:
                    del1,flag_positive,prev_dir=check_thesaurus(t1_ap_rule,t1_a_rule[0], word_derivation,t2_ap_rule, 1, flag_positive, prev_dir,kglr_settings.f_w_rule_t2_in_t1,data_w,"fwd",1)
                    del1,flag_positive,prev_dir=check_thesaurus(t2_ap_rule,word_derivation, t1_a_rule[0],t1_ap_rule, 2, flag_positive, prev_dir,kglr_settings.f_w_rule_t1_in_t2,data_w,"rev",1)
                
                if not flag_positive:
                    for word_derivation in derivation_t1:
                        #print "active-passive 2",t1_L_rule,t2_L_rule
                        del1,flag_positive,prev_dir=check_thesaurus(t1_ap_rule,word_derivation,t2_a_rule[0],t2_ap_rule, 1, flag_positive, prev_dir,kglr_settings.f_w_rule_t2_in_t1,data_w,"fwd",1)
                        del1,flag_positive,prev_dir=check_thesaurus(t2_ap_rule,t2_a_rule[0],word_derivation,t1_ap_rule, 2, flag_positive, prev_dir,kglr_settings.f_w_rule_t1_in_t2,data_w,"rev",1)
            #end: termination condition check (type2)
         
        ##Branch 3.10: Active Passive                   
        if not flag_rev_gate and kglr_settings.rule_active_passive_be:
            t1_P_rule_ap=t1_P_rule
            t2_P_rule_ap=t2_P_rule
            t1_L_rule_ap=t1_L_rule
            t2_L_rule_ap=t2_L_rule
            prev_dir=0
            del1="False"
            t1_L_rule_ap,t2_L_rule_ap,t1_P_rule_ap,t2_P_rule_ap,del1,flag_positive,prev_dir= rule_active_passive_be_fn("fwd",t1_L_rule_ap, t2_L_rule_ap, t1_P_rule_ap, t2_P_rule_ap, del1, flag_positive, prev_dir,data_w)
            t2_L_rule_ap, t1_L_rule_ap, t2_P_rule_ap, t1_P_rule_ap, del1, flag_positive, prev_dir=rule_active_passive_be_fn("rev",t2_L_rule_ap, t1_L_rule_ap, t2_P_rule_ap, t1_P_rule_ap, del1, flag_positive, prev_dir,data_w)
            
        if not flag_positive:
            t1_P_rule,t1_L_rule,flag_found=check_verbverb_construction(t1_P_rule_lightverbDT,t1_L_rule_lightverbDT,t2_L_rule_lightverbDT)
        if not flag_positive:
            prev_dir=1
            if t1_L_rule==t2_L_rule:
                del1,flag_positive,prev_dir=check_thesaurus(t1_P_rule,t1_L_rule,t2_L_rule,t2_P_rule, 2, flag_positive, prev_dir,kglr_settings.f_w_rule_t2_in_t1,data_w,"fwd",1)           
            
        ##Branch 4##
        ## Rule 4.1: Wordnet Hypernyms
        if not flag_positive and kglr_settings.flag_wordnet_hypernym_hyponym:
            #regret@R@;feel@R@
            prev_dir=0
            kglr_settings.flag_affix_call=1
            t1_affix_wn,t2_affix_wn,t1_P_affix_wn,t2_P_affix_wn=remove_common_affix(t1_L, t2_L,t1_P,t2_P)
            if kglr_settings.rules_dt:
                if len(t1_affix_wn)>0:
                    t1_P_affix_wn,t1_affix_wn=strip_DT(t1_P_affix_wn,t1_affix_wn)
                if len(t2_affix_wn):
                    t2_P_affix_wn,t2_affix_wn=strip_DT(t2_P_affix_wn,t2_affix_wn)
            if t1_affix_wn and t2_affix_wn:
                t1_affix_wn="-".join(t1_affix_wn)
                t2_affix_wn="-".join(t2_affix_wn)
                flag_hypernym, flag_hyponym,flag_common_hypernym=check_relation(t1_P_affix_wn,t1_affix_wn, t2_affix_wn,t2_P_affix_wn, 1, flag_positive, prev_dir,kglr_settings.f_w_affix_t2_in_t1,data_w,"fwd",1)
                if flag_hypernym or flag_hyponym or flag_common_hypernym:
                    kglr_settings.f_w_rule_t2_in_t1.write(data_w)
                    flag_positive=1
                kglr_settings.f_w.write(del1+" , ")
                kglr_settings.f_w.write(del1+" , ")
                kglr_settings.flag_affix_call=0    
                
        #Branch 5
        if not flag_positive and kglr_settings.flag_affix:
            prev_dir=0
            kglr_settings.flag_affix_call=1
            t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L, t2_L,t1_P,t2_P)
            if kglr_settings.rules_dt:
                if len(t1_affix)>0:
                    t1_P_affix,t1_affix=strip_DT(t1_P_affix,t1_affix)
                if len(t2_affix):
                    t2_P_affix,t2_affix=strip_DT(t2_P_affix,t2_affix)
            
            if not (t1_affix) and not(t2_affix):
                flag_positive=1
                kglr_settings.f_w_affix_t1_in_t2.write(line_w+"\n")
            else:
                t1_affix=" ".join(t1_affix)
                t2_affix=" ".join(t2_affix)
                del1,flag_positive,prev_dir=check_thesaurus(t1_P_affix,t1_affix, t2_affix,t2_P_affix, 1, flag_positive, prev_dir,kglr_settings.f_w_affix_t2_in_t1,data_w,"fwd",1)
                kglr_settings.f_w.write(del1+" , ")
                del1,flag_positive,prev_dir=check_thesaurus(t2_P_affix,t2_affix, t1_affix,t1_P_affix, 2, flag_positive, prev_dir,kglr_settings.f_w_affix_t1_in_t2,data_w,"rev",1)
                kglr_settings.f_w.write(del1+" , ")
            kglr_settings.flag_affix_call=0

        #Branch 6
        #Rule Redundant preposition
        if not flag_positive and kglr_settings.flag_affix and kglr_settings.rule_preposition_null:#redundant preposition
            prev_dir=0
            kglr_settings.flag_affix_call=1
            t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L, t2_L,t1_P,t2_P)
            if kglr_settings.rules_dt:
                if len(t1_affix)>0:
                    t1_P_affix,t1_affix=strip_DT_deep(t1_P_affix,t1_affix)
                if len(t2_affix):
                    t2_P_affix,t2_affix=strip_DT(t2_P_affix,t2_affix)
            
            if not (t1_affix) and not(t2_affix):
                flag_positive=1
                kglr_settings.f_w_affix_t1_in_t2.write(line_w+"\n")
            else:
                t1_affix=" ".join(t1_affix)
                t2_affix=" ".join(t2_affix)
            
                t1_affix_pn=[]
                t2_affix_pn=[]
                
                t1_affix_pn_P=[]
                t2_affix_pn_P=[]
                
                if kglr_settings.rule_preposition_null and t1_affix and t2_affix:
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
                            
                    del1,flag_positive,prev_dir=check_thesaurus(t1_P_affix,t1_affix, t2_affix,t2_P_affix, 1, flag_positive, prev_dir,kglr_settings.f_w_affix_t2_in_t1,data_w,"fwd",1)
                    kglr_settings.f_w.write(del1+" , ")
                    del1,flag_positive,prev_dir=check_thesaurus(t2_P_affix,t2_affix, t1_affix,t1_P_affix, 2, flag_positive, prev_dir,kglr_settings.f_w_affix_t1_in_t2,data_w,"rev",1)
                    kglr_settings.f_w.write(del1+" , ")
            kglr_settings.flag_affix_call=0
            
        #Branch 7
        if kglr_settings.flag_deep_dt:
            t1_affix,t2_affix,t1_P_affix,t2_P_affix=remove_common_affix(t1_L, t2_L,t1_P,t2_P)
            if kglr_settings.rules_dt:
                if len(t1_affix)>0:
                    t1_P_affix,t1_affix=strip_DT_deep(t1_P_affix,t1_affix)
                if len(t2_affix):
                    t2_P_affix,t2_affix=strip_DT(t2_P_affix,t2_affix)
            
            if not (t1_affix) and not(t2_affix) and not flag_positive:
                flag_positive=1
                kglr_settings.f_w_affix_t1_in_t2.write(line_w+"\n")
        
        #Branch 8: Rule: Deverbal noun
        if kglr_settings.flag_noun_verb_wn and not flag_positive:
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
                        kglr_settings.f_w_affix_t1_in_t2.write(line_w+"\n")
                        
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
                        kglr_settings.f_w_affix_t1_in_t2.write(line_w+"\n")
                            
        kglr_settings.f_w.write("\n")
        if flag_positive==0 and tag=="1":
            kglr_settings.f_w_not_handled_1.write(line+" , "+";".join(t1_P)+";".join(t2_P)+"\n")
            
        if flag_positive==1:
                flag_positive=0
                if tag=="0":
                    kglr_settings.fp=kglr_settings.fp+1
                    kglr_settings.f_score.write(data_w)
                elif tag=="1":
                    kglr_settings.tp=kglr_settings.tp+1
                elif tag=="2":
                    kglr_settings.tp_2+=1
        
        kglr_settings.f_w.flush()
    
        kglr_settings.f_w_not_handled_1.flush()
        
        kglr_settings.f_w_rule_t2_in_t1.flush()
        kglr_settings.f_w_rule_t1_in_t2.flush()
        
        kglr_settings.f_w_whole_t2_in_t1.flush()
        kglr_settings.f_w_whole_t1_in_t2.flush()
        
        kglr_settings.f_w_affix_t2_in_t1.flush()
        kglr_settings.f_w_affix_t1_in_t2.flush()    

    try:
        print "tp_2",kglr_settings.tp_2
        print "tp",kglr_settings.tp
        print "fp",kglr_settings.fp
        print "all_p",kglr_settings.all_p
        print "all_p_2",kglr_settings.all_p_2
        precision=float(kglr_settings.tp)/float(kglr_settings.tp+kglr_settings.fp)
        precision_naacl = float(kglr_settings.tp+kglr_settings.tp_2)/float(kglr_settings.tp+kglr_settings.tp_2+kglr_settings.fp)
        print "precision",float(kglr_settings.tp)/float(kglr_settings.tp+kglr_settings.fp)
        print "precision naacl",float(kglr_settings.tp+kglr_settings.tp_2)/float(kglr_settings.tp+kglr_settings.tp_2+kglr_settings.fp)
        recall=float(kglr_settings.tp)/float(kglr_settings.all_p)
        print "recall",float(kglr_settings.tp)/float(kglr_settings.all_p)
        recall_naacl  =float(kglr_settings.tp+kglr_settings.tp_2)/float(kglr_settings.all_p+kglr_settings.all_p_2)
        print "recall naacl",float(kglr_settings.tp+kglr_settings.tp_2)/float(kglr_settings.all_p+kglr_settings.all_p_2)
        F_score=2*precision*recall/(precision+recall)
        print "F-score",F_score
        F_score_naacl=2*precision_naacl*recall_naacl/(precision_naacl+recall_naacl)
        print "F-score naacl",F_score_naacl
        kglr_settings.f_score.write("tp "+str(kglr_settings.tp)+"\n")
        kglr_settings.f_score.write("fp "+str(kglr_settings.fp)+"\n")
        kglr_settings.f_score.write("all_p "+str(kglr_settings.all_p)+"\n")
        kglr_settings.f_score.write("precision "+str(precision)+"\n")
        kglr_settings.f_score.write("recall "+str(recall)+"\n")
        kglr_settings.f_score.write("F_score "+str(F_score)+"\n")
    except:
        print "error"
    
    kglr_settings.f_w_not_handled_1.close()
    kglr_settings.f_w_whole_t2_in_t1.close()
    kglr_settings.f_w_whole_t1_in_t2.close()
    kglr_settings.f_w_affix_t2_in_t1.close()
    kglr_settings.f_w_affix_t1_in_t2.close()
    kglr_settings.f_score.close()
kglr_settings.f_w.close()
kglr_settings.f_log.close()
