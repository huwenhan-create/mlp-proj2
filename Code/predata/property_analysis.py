import numpy as np
import pandas as pd

def property_analysis(target:str,df):
    tar = df[target]
    freq = tar.value_counts()/tar.count()
    return freq