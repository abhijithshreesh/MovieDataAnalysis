import os
import pandas as pd
import logging
from config_parser import ParseConfig
from data_extractor import ExtractData
from collections import Counter
import math
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

conf = ParseConfig()

class ActorTag(object):

    def __init__(self):
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = ExtractData(self.data_set_loc)

    def assign_idf_weight(self, data_frame, unique_tags):
        idf_counter = {tag: 0 for tag in unique_tags}
        data_frame.tag = pd.Series([set(tags.split(',')) for tags in data_frame.tag], index=data_frame.index)
        for tag_list in data_frame.tag:
            for tag in tag_list:
                idf_counter[tag] += 1
        for tag, count in list(idf_counter.items()):
            idf_counter[tag] = math.log(len(data_frame.index)/count)
        return idf_counter

    def assign_tf_weight(self, tag_series):
        counter = Counter()
        for each in tag_series:
            counter[each] += 1
        total = sum(counter.values())
        for each in counter:
            counter[each] = (counter[each]/total)
        return dict(counter)

    def assign_rank_weight(self, data_frame):
        groupby_movies = data_frame.groupby("movieid")
        movie_rank_weight_dict = {}
        for movieid, info_df in groupby_movies:
           max_rank = info_df.actor_movie_rank.max()
           for rank in info_df.actor_movie_rank.unique():
             movie_rank_weight_dict[(movieid, rank)] = (max_rank - rank + 1)/max_rank*100
        return movie_rank_weight_dict

    def get_model_weight(self, tf_weight_dict, idf_weight_dict, rank_weight_dict, tag_df, model):
        if model == "TF":
            tag_df["value"] = pd.Series(
                [(ts_weight + (tf_weight_dict.get(movieid, 0).get(tag, 0)*100) + rank_weight_dict.get((movieid, rank), 0)) for
                 index, ts_weight, tag, movieid, rank
                 in zip(tag_df.index, tag_df.timestamp_weight, tag_df.tag, tag_df.movieid, tag_df.actor_movie_rank)],
                index=tag_df.index)
        else:
            tag_df["value"] = pd.Series(
                [(ts_weight + (tf_weight_dict.get(movieid, 0).get(tag, 0)*(idf_weight_dict.get(tag, 0))*100) + rank_weight_dict.get((movieid, rank), 0)) for
                 index, ts_weight, tag, movieid, rank
                 in zip(tag_df.index, tag_df.timestamp_weight, tag_df.tag, tag_df.movieid, tag_df.actor_movie_rank)],
                index=tag_df.index)
        return tag_df

    def combine_computed_weights(self, data_frame, actor_tag_dict, rank_weight_dict, model):
        groupby_actor_dict = data_frame.groupby("name")
        for actorid, tag_df in groupby_actor_dict:
                    tag_df = tag_df.reset_index()
                    unique_tags = tag_df.tag.unique()
                    temp_df = tag_df.groupby(['movieid'])['tag'].apply(lambda x: ','.join(x)).reset_index()
                    movie_tag_dict = dict(zip(temp_df.movieid, temp_df.tag))
                    tf_weight_dict = {movie: self.assign_tf_weight(tags.split(',')) for movie, tags in
                                      list(movie_tag_dict.items())}
                    idf_weight_dict = {}
                    if model != 'TF':
                        idf_weight_dict = self.assign_idf_weight(temp_df, unique_tags)
                    tag_df = self.get_model_weight(tf_weight_dict, idf_weight_dict, rank_weight_dict, tag_df, model)
                    tag_df["total"] = tag_df.groupby(['tag'])['value'].transform('sum')
                    tag_df = tag_df.drop_duplicates("tag").sort_values("total", ascending=False)
                    actor_tag_dict[actorid] = dict(zip(tag_df.tag, tag_df.total))
        return actor_tag_dict

    def assign_weight(self, data_frame, model):
        data_frame = data_frame[data_frame['timestamp'].notnull()]
        data_frame = data_frame.drop(["userid"], axis=1)
        rank_weight_dict = self.assign_rank_weight(data_frame[['movieid', 'actor_movie_rank']])
        data_frame = data_frame.sort_values("timestamp", ascending=True)
        data_frame_len = len(data_frame.index)
        data_frame["timestamp_weight"] = pd.Series([(index+1)/data_frame_len*10 for index in data_frame.index], index= data_frame.index)
        actor_tag_dict = self.combine_computed_weights(data_frame, {}, rank_weight_dict, model)
        actor_tag_data_frame = pd.DataFrame.from_dict(list(actor_tag_dict.items()))
        return actor_tag_data_frame

    def merge_movie_actor_and_tag(self, model):
        mov_act = self.data_extractor.get_movie_actor_data()
        ml_tag = self.data_extractor.get_ml_tags_data()
        genome_tag = self.data_extractor.get_genome_tags_data()
        actor_info = self.data_extractor.get_imdb_actor_info()
        actor_movie_info = mov_act.merge(actor_info, how="left", on="actorid")
        tag_data_frame = ml_tag.merge(genome_tag, how="left", left_on="tagid", right_on="tagId")
        merged_data_frame = actor_movie_info.merge(tag_data_frame, how="left", on="movieid")
        actor_tag_data_frame = self.assign_weight(merged_data_frame, model)
        actor_tag_data_frame.columns = ['actor_name', 'tags']
        actor_tag_data_frame.to_csv(os.path.join(self.data_set_loc, "ACT_TAG_%s.csv" % model), index=False)


if __name__ == "__main__":
    obj = ActorTag()
    obj.merge_movie_actor_and_tag(model='TFIDF')

