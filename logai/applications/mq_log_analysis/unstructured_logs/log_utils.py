#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#


import re 
import pandas as pd 

def clean_and_truncate(loglines: pd.Series):
    """Method to clean and truncate logline 

    Args:
        loglines (pd.Series): log data to be cleaned

    Returns:
        pd.Series : cleaned log data 
    """
    loglines = loglines.apply(lambda x: x.replace('XX','').replace('*','').replace('\[[0-9., ]*\]',''))
    loglines = loglines.apply(lambda x: ' '.join(x[:500].split(' ')[:100]) if len(x)>1000 else x)
    return loglines


def tokenize_logline(logline: str):
    """tokenize logline string

    Args:
        logline (str): string to be tokenized

    Returns:
        str: whitespace separated token sequence output as string 
    """
    logline = logline.replace('sfdc.common.messaging.','').replace('common.messaging.','')
    funcname = logline.split(' ')[0]
    logline = ' '.join(logline.split(' ')[1:]).replace(funcname, '').replace('*','')
    funcname_tokenized = re.sub(r'(?<!^)(?=[A-Z])', ' ', funcname).replace('.',' ').lower()
    logline =  re.sub(' +', ' ', funcname_tokenized+' '+logline)[:100].strip()
    return logline 


