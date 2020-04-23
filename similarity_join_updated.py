# similarity_join.py
"""
This code solve the Entity Resolution (ER) problem by implementing 4 sequential steps:
1) Data Preprocessing - (function: preprocess_df) Tokenizing the required column and returning a processed
                          dataframe with a column (joinKey) having list of token
2) Data Filtering - (function: filtering) Filtering non matching pairs between two dataframes and returning a
                      dataframe (cand_df) with joined results based on atleast one matching token in joinKey columns
3) Jaccard Similarity - (function: verification) Computing Jaccard similarity of two sets (R & S) by calculating
                          (R n S)/(R u S) and then compare its value with a given threshold for matching pairs
4) Evaluating ER Result - (function: evaluate) Compute precision, recall, and fmeasure of $result based on $ground_truth
                            $ground_truth is a list of matching pairs labeld by humans, results list of matching pairs
                            identified by ER algorithm
"""
import re
import pandas as pd


class SimilarityJoin:
    def __init__(self, data_file1, data_file2):
        self.df1 = pd.read_csv(data_file1)
        self.df2 = pd.read_csv(data_file2)

    # Tokenizing given columns using regular expression, removing Null entries,
    # and returning homogeneous lower case data
    def preprocess_df(self, df, cols):
        df['concatenated_columns'] = df[cols[0]].fillna('') + ' ' + df[cols[1]].fillna('')
        df['concatenated_columns'] = df['concatenated_columns'].str.lower()
        tokenized_list = []
        for details in df['concatenated_columns']:
            tokenized_data = re.split(r'\W+', details)
            tokenized_data = list(filter(None, tokenized_data))
            tokenized_list.append(tokenized_data)

        df['joinKey'] = tokenized_list
        del df['concatenated_columns']
        return df

    # Removing non matching pairs by implementing inner join on two dataframes
    # and returning a new dataframe with matching join results
    def filtering(self, df1, df2):
        df1_temp = df1.explode('joinKey')
        df2_temp = df2.explode('joinKey')
        #df1_temp.columns = ['id1', 'title1', 'description1', 'manufacturer1', 'price1', 'joinKey1']
        #df2_temp.columns = ['id2', 'name2', 'description2', 'manufacturer2', 'price2', 'joinKey2']
        df1_temp = df1_temp.rename(columns={'joinKey': 'joinKey1', 'id':'id1'})
        df2_temp = df2_temp.rename(columns={'joinKey': 'joinKey2', 'id':'id2'})
        df_final = pd.merge(df1_temp, df2_temp, left_on=df1_temp['joinKey1'], right_on=df2_temp['joinKey2'],
                            how='inner')
        df_final = df_final.drop_duplicates(subset=['id1', 'id2'])
        df_final = df_final[['id1', 'joinKey1', 'id2', 'joinKey2']]
        df_final = pd.merge(df_final, df1, left_on=df_final['id1'], right_on=df1['id'], how='inner')
        df_final = df_final[['id1', 'joinKey', 'id2', 'joinKey2']]
        df_final.rename(columns={'joinKey': 'joinKey1'}, inplace=True)
        df_final = pd.merge(df_final, df2, left_on=df_final['id2'], right_on=df2['id'], how='inner')
        df_final = df_final[['id1', 'joinKey1', 'id2', 'joinKey']]
        df_final.rename(columns={'joinKey': 'joinKey2'}, inplace=True)
        return df_final

    # Computing jaccard value and comparing it with the threshold to return similar pairs
    def verification(self, cand_df, threshold):
        # Function return a float value(Jaccard) computed from the given sets/lists
        def jaccard_similarity(list1, list2):
            set1 = set(list1)
            set2 = set(list2)
            numerator = len(set1.intersection(set2))
            denominator = len(set1.union(set2))
            jaccard = float(numerator / denominator)
            return jaccard

        # Adding a jaccard column to dataframe having jaccard similarity value
        cand_df['jaccard'] = cand_df.apply(lambda x: jaccard_similarity(x.joinKey1, x.joinKey2), axis=1)
        cand_df = cand_df[['id1', 'id2', 'jaccard']]
        # Filtering out non matching pairs
        cand_df = cand_df.loc[cand_df['jaccard'] >= threshold]
        return cand_df

    # Evaluating the result (ER Algorithm generated) with ground_truth
    def evaluate(self, result, ground_truth):
        # Converting tokenized list column to a set of individual tokens
        result_set = set([item for sublist in result for item in sublist])
        ground_truth_set = set([item for sublist in ground_truth for item in sublist])

        # Calculating precision, recall, fmeasure using their formulas
        precision = len(result_set.intersection(ground_truth_set)) / len(result_set)
        recall = len(result_set) / len(ground_truth_set)
        fmeasure = ((2 * precision * recall) / (precision + recall))
        return (precision, recall, fmeasure)

    # Function calls and insights into data processing
    def jaccard_join(self, cols1, cols2, threshold):
        new_df1 = self.preprocess_df(self.df1, cols1)
        new_df2 = self.preprocess_df(self.df2, cols2)
        print("Before filtering: %d pairs in total" % (self.df1.shape[0] * self.df2.shape[0]))

        cand_df = self.filtering(new_df1, new_df2)
        print("After Filtering: %d pairs left" % (cand_df.shape[0]))

        result_df = self.verification(cand_df, threshold)
        print("After Verification: %d similar pairs" % (result_df.shape[0]))

        return result_df


if __name__ == "__main__":
    er = SimilarityJoin("Amazon.csv", "Google.csv")
    amazon_cols = ["title", "manufacturer"]
    google_cols = ["name", "manufacturer"]
    result_df = er.jaccard_join(amazon_cols, google_cols, 0.5)

    result = result_df[['id1', 'id2']].values.tolist()
    ground_truth = pd.read_csv("Amazon_Google_perfectMapping.csv").values.tolist()
    print("(precision, recall, fmeasure) = ", er.evaluate(result, ground_truth))
