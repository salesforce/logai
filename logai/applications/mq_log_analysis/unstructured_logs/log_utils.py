#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#


import re 

def clean_and_truncate(preprocessed_loglines):
    print ('unstances with nan ',len([str(x) for x in preprocessed_loglines if type(x)==float]))
    print (len(preprocessed_loglines))
    preprocessed_loglines = preprocessed_loglines.apply(lambda x: x.replace('XX','').replace('*','').replace('\[[0-9., ]*\]',''))
    preprocessed_loglines = preprocessed_loglines.apply(lambda x: ' '.join(x[:500].split(' ')[:100]) if len(x)>1000 else x)
    return preprocessed_loglines

def tokenize_logline(logline):
    logline = logline.replace('sfdc.common.messaging.','').replace('common.messaging.','')
    funcname = logline.split(' ')[0]
    logline = ' '.join(logline.split(' ')[1:]).replace(funcname, '').replace('*','')
    funcname_tokenized = re.sub(r'(?<!^)(?=[A-Z])', ' ', funcname).replace('.',' ').lower()
    logline =  re.sub(' +', ' ', funcname_tokenized+' '+logline)[:100].strip()
    return logline 


