import os
import pickle
import pandas as pd
from pandas.api.types import is_numeric_dtype
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import joblib

# download nltk resources
nltk.download('stopwords')
nltk.download('punkt')

this_dir, this_filename = os.path.split(__file__)  # Get path of data.pkl
rf_Filename = os.path.join(this_dir, 'resources/RandForest.pkl')
with open(rf_Filename, 'rb') as file:  Pickled_LR_Model = pickle.load(file)

del_pattern = r'([^,;\|]+[,;\|]{1}[^,;\|]+){1,}'
del_reg = re.compile(del_pattern)

delimeters = r"(,|;|\|)"
delimeters = re.compile(delimeters)

url_pat = r"(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
url_reg = re.compile(url_pat)

email_pat = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,6}\b"
email_reg = re.compile(email_pat)

stop_words = set(stopwords.words('english'))



def summary_stats(dat, key_s):
    b_data = []
    for col in key_s:
        nans = np.count_nonzero(pd.isnull(dat[col]))
        dist_val = len(pd.unique(dat[col].dropna()))
        Total_val = len(dat[col])
        mean = 0
        std_dev = 0
        var = 0
        min_val = 0
        max_val = 0
        if is_numeric_dtype(dat[col]):
            mean = np.mean(dat[col])

            if pd.isnull(mean):
                mean = 0
                std_dev = 0
                #var = 0
                min_val = 0
                max_val = 0
            else:
                std_dev = np.std(dat[col])
                var = np.var(dat[col])
                min_val = float(np.min(dat[col]))
                max_val = float(np.max(dat[col]))
        b_data.append([Total_val, nans, dist_val, mean, std_dev, min_val, max_val])
    return b_data

def castability_feature(dat, column_names):
    castability_list = []
    #make sure the value you are avaluating is not nan
    for keys in column_names:
        #print(keys)
        i = 0
        while pd.isnull(dat[keys][i]):
            i += 1
            if i > len(dat[keys]) - 2:
                break
        #if type is string try casting
        if dat[keys][i].__class__.__name__ == 'str':
            try:
                castability = str(type(eval(dat[keys][i])))
                castability_list.append(1)
            except:
                castability_list.append(0)
        else:
            castability_list.append(0)
    return castability_list


def numeric_extraction(dat,column_names):
    #0 no , 1 yes
    numeric_extraction_list = []
    #make sure the value you are avaluating is not nan
    for keys in column_names:
        i = 0
        while pd.isnull(dat[keys][i]):
            i += 1
            if i > len(dat[keys]) - 2:
                break
        val = 0

        if dat[keys][i].__class__.__name__ == 'str':
            #print('yes')
            #check whether any number can be extracted
            try:
                #it will faile when you have no numbers or if you have two numbers seperated by space
                float(re.sub('[^0-9\. ]', ' ',dat[keys][i]))
                #print('yes')
                val = 1
            except:
                pass

        numeric_extraction_list.append(val)

    return numeric_extraction_list

def get_sample(dat, key_s):
    rand = []
    for name in key_s: # TODO Omg this is bad. Should use key_s.
        rand_sample = list(pd.unique(dat[name]))
        rand_sample = rand_sample[:5]
        while len(rand_sample) < 5:
            rand_sample.append(list(pd.unique(dat[name]))[np.random.randint(len(list(pd.unique(dat[name]))))])
        rand.append(rand_sample[:5])
    return rand

def get_avg_tokens(samples):

    # samples contain list of length len(keys) of 5-sample list.
    avg_tokens = []
    for sample_list in samples:
        list_of_num_tokens = [len(str(sample).split()) for sample in sample_list]
        avg_tokens.append(sum(list_of_num_tokens) / len(list_of_num_tokens))

    return avg_tokens

# summary_stat_result has a structure like [[Total_val, nans, dist_va, ...], ...].
def get_ratio_dist_val(summary_stat_result):
    ratio_dist_val = []
    for r in summary_stat_result:
        ratio_dist_val.append(r[2]*100.0 / r[0])
    return ratio_dist_val

def get_ratio_nans(summary_stat_result):
    ratio_nans = []
    for r in summary_stat_result:
        ratio_nans.append(r[1]*100.0 / r[0])
    return ratio_nans




# y = df['out/in']

def FeaturizeFile(df):
	# df = pd.read_csv(CSVfile,encoding = 'latin1')

	stats = []
	attribute_name = []
	sample = []
	id_value = []
	i = 0

	castability = []
	number_extraction = []

	avg_tokens = []
	ratio_dist_val = []
	ratio_nans = []

	keys = list(df.keys())

	attribute_name.extend(keys)
	summary_stat_result = summary_stats(df, keys)
	stats.extend(summary_stat_result)
	samples = get_sample(df,keys)
	sample.extend(samples)


	# castability.extend(castability_feature(df, keys))
	# number_extraction.extend(numeric_extraction(df, keys))

	# avg_tokens.extend(get_avg_tokens(samples))
	ratio_dist_val.extend(get_ratio_dist_val(summary_stat_result))
	ratio_nans.extend(get_ratio_nans(summary_stat_result))


	csv_names = ['Attribute_name', 'total_vals', 'num_nans', 'num_of_dist_val', 'mean', 'std_dev', 'min_val',
	             'max_val', '%_dist_val', '%_nans', 'sample_1', 'sample_2', 'sample_3','sample_4','sample_5'
	            ]
	golden_data = pd.DataFrame(columns = csv_names)

	for i in range(len(attribute_name)):
	    # print(attribute_name[i])
	    val_append = []
	    val_append.append(attribute_name[i])
	    val_append.extend(stats[i])

	    val_append.append(ratio_dist_val[i])
	    val_append.append(ratio_nans[i])

	    val_append.extend(sample[i])
	#     val_append.append(castability[i])
	#     val_append.append(number_extraction[i])
	#     val_append.append(avg_tokens[i])

	    golden_data.loc[i] = val_append
	#     print(golden_data)


	curdf = golden_data

	for row in curdf.itertuples():

	    # print(row[11])
	    is_list = False
	    curlst = [row[11],row[12],row[13],row[14],row[15]]

	    delim_cnt,url_cnt,email_cnt,date_cnt =0,0,0,0
	    chars_totals,word_totals,stopwords,whitespaces,delims_count = [],[],[],[],[]

	    for value in curlst:
	        word_totals.append(len(str(value).split(' ')))
	        chars_totals.append(len(str(value)))
	        whitespaces.append(str(value).count(' '))

	        if del_reg.match(str(value)):  delim_cnt += 1
	        if url_reg.match(str(value)):  url_cnt += 1
	        if email_reg.match(str(value)):  email_cnt += 1

	        delims_count.append(len(delimeters.findall(str(value))))

	        tokenized = word_tokenize(str(value))
	        # print(tokenized)
	        stopwords.append(len([w for w in tokenized if w in stop_words]))

	        try:
	            _ = pd.Timestamp(value)
	            date_cnt += 1
	        except ValueError: date_cnt += 0

	    # print(delim_cnt,url_cnt,email_cnt)
	    if delim_cnt > 2:  curdf.at[row.Index, 'has_delimiters'] = True
	    else: curdf.at[row.Index, 'has_delimiters'] = False

	    if url_cnt > 2:  curdf.at[row.Index, 'has_url'] = True
	    else: curdf.at[row.Index, 'has_url'] = False

	    if email_cnt > 2:  curdf.at[row.Index, 'has_email'] = True
	    else: curdf.at[row.Index, 'has_email'] = False

	    if date_cnt > 2:  curdf.at[row.Index, 'has_date'] = True
	    else: curdf.at[row.Index, 'has_date'] = False

	    curdf.at[row.Index, 'mean_word_count'] = np.mean(word_totals)
	    curdf.at[row.Index, 'std_dev_word_count'] = np.std(word_totals)

	    curdf.at[row.Index, 'mean_stopword_total'] = np.mean(stopwords)
	    curdf.at[row.Index, 'stdev_stopword_total'] = np.std(stopwords)

	    curdf.at[row.Index, 'mean_char_count'] = np.mean(chars_totals)
	    curdf.at[row.Index, 'stdev_char_count'] = np.std(chars_totals)

	    curdf.at[row.Index, 'mean_whitespace_count'] = np.mean(whitespaces)
	    curdf.at[row.Index, 'stdev_whitespace_count'] = np.std(whitespaces)

	    curdf.at[row.Index, 'mean_delim_count'] = np.mean(whitespaces)
	    curdf.at[row.Index, 'stdev_delim_count'] = np.std(whitespaces)

	    if curdf.at[row.Index, 'has_delimiters'] and curdf.at[row.Index, 'mean_char_count'] < 100: curdf.at[row.Index, 'is_list'] = True
	    else: curdf.at[row.Index, 'is_list'] = False

	    if curdf.at[row.Index, 'mean_word_count'] > 10: curdf.at[row.Index, 'is_long_sentence'] = True
	    else: curdf.at[row.Index, 'is_long_sentence'] = False

	    # print(np.mean(stopwords))

	    # print('\n\n\n')

	golden_data = curdf

	return golden_data


# vectorizer,vectorizer1,vectorizer2 = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("vectorizer.pkl", "rb"))),CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("vectorizer1.pkl", "rb"))),CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("vectorizer2.pkl", "rb")))
vectorizerName = joblib.load(os.path.join(this_dir, "resources/dictionaryName.pkl"))
vectorizerSample = joblib.load(os.path.join(this_dir, "resources/dictionarySample.pkl"))

def FeatureExtraction(data):

    data1 = data[['total_vals', 'num_nans', '%_nans', 'num_of_dist_val', '%_dist_val', 'mean', 'std_dev', 'min_val', 'max_val','has_delimiters', 'has_url', 'has_email', 'has_date', 'mean_word_count',
       'std_dev_word_count', 'mean_stopword_total', 'stdev_stopword_total',
       'mean_char_count', 'stdev_char_count', 'mean_whitespace_count',
       'stdev_whitespace_count', 'mean_delim_count', 'stdev_delim_count',
       'is_list', 'is_long_sentence']]
    data1 = data1.reset_index(drop=True)
    data1 = data1.fillna(0)

    arr = data['Attribute_name'].values
    arr = [str(x) for x in arr]

    arr1 = data['sample_1'].values
    arr1 = [str(x) for x in arr1]
    arr2 = data['sample_2'].values
    arr2 = [str(x) for x in arr2]
    arr3 = data['sample_3'].values
    arr3 = [str(x) for x in arr3]
    print(len(arr1),len(arr2))

    X = vectorizerName.transform(arr)
    X1 = vectorizerSample.transform(arr1)
    X2 = vectorizerSample.transform(arr2)

    attr_df = pd.DataFrame(X.toarray())
    sample1_df = pd.DataFrame(X1.toarray())
    sample2_df = pd.DataFrame(X2.toarray())
    print(len(data1),len(attr_df),len(sample1_df),len(sample2_df))

    data2 = pd.concat([data1, attr_df], axis=1, sort=False)
    print(len(data2))
    return data2


def Load_RF(df):
	y_RF = Pickled_LR_Model.predict(df).tolist()
	return y_RF
