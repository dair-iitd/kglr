'''
Created on May 20, 2016

@author: Prachi
'''
import re,sys,os,nltk,itertools
import MySQLdb as mdb
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.tag.stanford import StanfordPOSTagger

def init():
    ###Set paths###
    global java_path;java_path = "C:\\Program Files\\Java\\jdk1.8.0_60\\bin\\java.exe"
    global postagger_model_path;postagger_model_path='E:\\nltk_data\\stanford\\stanford-postagger.jar'
    global distsimtagger_model_path;distsimtagger_model_path='E:\\nltk_data\\stanford\\model\\english-bidirectional-distsim.tagger'
    global path;path=""#project_path
    global rule_in_file;rule_in_file="in_files\\ablation.txt"#input file
    #global rule_out_file;rule_out_file=""#output file
    #Data files#
    global redundant_prep_file;redundant_prep_file="data\\ch9out2-r-0000all_norm_IN_null.txt"
    global prep_syn_file;prep_syn_file="data\\prep_pair_all_freq.txt"
    global java_options_mem;java_options_mem='-mx2G'
    #thesaurus db#
    global thesaurus_db;thesaurus_db = 'thesaurus'
    global password_db;password_db=''
    ###Feature flags###
    #Thesaurus Synonyms
    global flag_whole;flag_whole=1
    global flag_antonyms;flag_antonyms=1
    global flag_syn_of_syn;flag_syn_of_syn=1
    #Negating rules
    global flag_not;flag_not=1
    #Wordnet Hypernyms
    global flag_wordnet_hypernym_hyponym;flag_wordnet_hypernym_hyponym=1
    global flag_level2_hypernyms;flag_level2_hypernyms=1
    #Dropping Modifiers
    global rule_superlative;rule_superlative=1
    global rule_JJ_NN_mod;rule_JJ_NN_mod=1
    #Gerund-Infinitive Equivalence
    global flag_gerund_infi_to;flag_gerund_infi_to=1
    #Deverbal Nouns
    global flag_noun_verb_wn;flag_noun_verb_wn=1
    #Light Verbs and Serial Verbs
    global rule_lightverb_dt;rule_lightverb_dt=1
    global rule_verbverb;rule_verbverb=1
    #Preposition Synonyms
    global rule_equal_preposition;rule_equal_preposition=1
    global rule_equal_preposition_data_freq;rule_equal_preposition_data_freq=1
    #Be-words & Determiners
    global flag_deep_dt;flag_deep_dt=1
    global rules_dt;rules_dt=1#interim removal of dt
    global rule_be;rule_be=1
    #Active-Passive
    global rule_active_passive_be;rule_active_passive_be=1
    #Redundant Preposition
    global rule_preposition_null;rule_preposition_null=1#X;learn about;Y --> X;learn;Y
    #Extra#be - preposition#("be sick of@R@;tire of@R@"("be sick of@R@;tire of@R@")
    global rule_be_prep;rule_be_prep=1
    
    ####Utility Flags###
    #del
    global flag_rules;flag_rules=1
    
    global flag_stanford_pos;flag_stanford_pos=1
    
    global flag_affix;flag_affix=1
    #utility
    global count_t2int1_prev;count_t2int1_prev=0
    global flag_affix_call;flag_affix_call=0
    global flag_rev;flag_rev=1#to ensure: X t1 Y -> X t2 Y
    global flag_bidir_syn;flag_bidir_syn=1#flag to check t1 in t2'syn and t2 in t1's syn
    global flag_pos_match;flag_pos_match=1#ensure matching of same pos words
    global flag_derivation;flag_derivation=1#extract derivations of given word
    
    #extra
    global rule_have;rule_have=1
    global rule_have_JJ;rule_have_JJ=1
    
    #relation validation
    global flag_be_trail;flag_be_trail=1
    
    ####Init values###
    global tp_2;tp_2=0
    global tp;tp=0
    global fp;fp=0
    global all_p;all_p=0
    global all_p_2;all_p_2= 0
    global flag_positive;flag_positive=1
    
    ###Output file###
    global f_score;f_score=open(path+"out_files\\score.txt","w")#Final Score file
    global f_w;f_w=open(path+"out_files\\200_tncf_annotated_part2_features_in.csv","w")           
    global f_w_not_handled_1;f_w_not_handled_1=open(path+"out_files\\not_handled_1.csv","w")
    global f_w_rule_t2_in_t1;f_w_rule_t2_in_t1=open(path+"out_files\\f_w_rule_t2_in_t1.csv","w")
    global f_w_rule_t1_in_t2;f_w_rule_t1_in_t2=open(path+"out_files\\f_w_rule_t1_in_t2.csv","w")
    global f_w_whole_t2_in_t1;f_w_whole_t2_in_t1=open(path+"out_files\\whole_t2_in_t1.csv","w")
    global f_w_whole_t1_in_t2;f_w_whole_t1_in_t2=open(path+"out_files\\whole_t1_in_t2.csv","w")
    global f_w_affix_t2_in_t1;f_w_affix_t2_in_t1=open(path+"out_files\\affix_t2_in_t1.csv","w")
    global f_w_affix_t1_in_t2;f_w_affix_t1_in_t2=open(path+"out_files\\affix_t1_in_t2.csv","w")
    global f_log;f_log=open("log.txt","w")
    
    global check_thesaurus_100;check_thesaurus_100={}#for speedup
    global check_db_100;check_db_100={}#for speedup

    global light_verb_old;light_verb_old=["take", "have", "give", "do", "make"]
    global verbs_verb;verbs_verb=["has","have","be","is","were","are","was","had","being","began","am","following","having","do","does",
                "did","started","been","became","left","help","helped","get","keep","think","got","gets","include",
                "suggest","used","see","consider","means","try","start","included","lets","say","continued",
                "go","includes","becomes","begins","keeps","begin","starts","said"]#,"stop"
    global verb_verb_norm;verb_verb_norm=["begin","start","continue","say"]
    global all_verb_verb;all_verb_verb=verb_verb_norm+verbs_verb#used for checking serial verb constructions
    global L_gerund_infi_to;L_gerund_infi_to=["attempt","begin","bother","cease","continue","deserve","neglect","omit","permit","start","fear","intend","recommend","advice","allow","permit","encourage","forbid","choose"]
    global L_gerund_infi_to_t1;L_gerund_infi_to_t1=["like","love","prefer"]
    
    global search_str;search_str=""
    global morphy_tag; morphy_tag = {'NN':wn.NOUN,'JJ':wn.ADJ,'VB':wn.VERB,'RB':wn.ADV}
    
    os.environ['JAVAHOME'] = java_path
    global st;st=StanfordPOSTagger(distsimtagger_model_path,postagger_model_path, java_options=java_options_mem)
    
