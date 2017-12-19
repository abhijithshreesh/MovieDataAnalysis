import logging
from config_parser import ParseConfig
from task_2 import GenreTag
import pandas as pd
import operator
import math
logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)
conf = ParseConfig()

class GenreDifferentiator(GenreTag):

    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")

    def combine_computed_weights(self, data_frame_1, data_frame_2, model):
        tag_df_1 = data_frame_1.reset_index()
        tag_df_2 = data_frame_2.reset_index()
        unique_tags = list(tag_df_1.tag.unique()) + list(tag_df_2.tag.unique())
        temp_df_1 = tag_df_1.groupby(['movieid'])['tag'].apply(lambda x: ','.join(x)).reset_index()
        temp_df_2 = tag_df_2.groupby(['movieid'])['tag'].apply(lambda x: ','.join(x)).reset_index()
        movie_tag_dict = dict(zip(temp_df_1.movieid, temp_df_1.tag))
        tf_weight_dict = {movie: self.assign_tf_weight(tags.split(',')) for movie, tags in list(movie_tag_dict.items())}
        idf_weight_dict = {}
        if model != 'TF':
            idf_weight_dict = self.assign_idf_weight(pd.concat([temp_df_1, temp_df_2]).drop_duplicates(), unique_tags)
        tag_df = self.get_model_weight(tf_weight_dict, idf_weight_dict, tag_df_1, model)
        tag_df["total"] = tag_df.groupby(['tag'])['value'].transform('sum')
        tag_df = tag_df.drop_duplicates("tag").sort_values("total", ascending=False)
        actor_tag_dict = dict(zip(tag_df["tag"], tag_df["total"]))
        return actor_tag_dict

    def count_tags_in_movies(self, data_frame, unique_tags, model):
        counter = {tag: 0 for tag in unique_tags}
        data_frame.tag = pd.Series([set(tags.split(',')) for tags in data_frame.tag], index=data_frame.index)
        if model == "P2_DIFF":
            for unique_tag in unique_tags:
                for tag_list in data_frame.tag:
                    for tag in tag_list:
                        if unique_tag!=tag:
                            counter[unique_tag] += 1
        else:
            for tag_list in data_frame.tag:
                for tag in tag_list:
                    counter[tag] += 1
        return dict(counter)

    def differentiate_genre(self, genre_1, genre_2, model):
        genre_data_frame = self.get_genre_data()
        genre_1_data_frame = genre_data_frame[genre_data_frame.genre == genre_1]
        genre_2_data_frame = genre_data_frame[genre_data_frame.genre == genre_2]
        tag_dict_genre_1 = self.combine_computed_weights(genre_1_data_frame, genre_2_data_frame, model)
        tag_dict_genre_2 = self.combine_computed_weights(genre_2_data_frame, genre_1_data_frame, model)
        tag_diff = {tag: float(tag_dict_genre_1.get(tag, 0)) - float(tag_dict_genre_2.get(tag, 0)) for tag in tag_dict_genre_1.keys()}
        tag_tuple = sorted(tag_diff.items(), key=operator.itemgetter(1), reverse=True)
        print({genre_1: dict(tag_tuple)})

    def compute_p1_weights(self, r, m , R, M):
        if r == m:
            return 0
        try:
            result = math.log(abs((((r+ 0.5)/(R-r + 1)) )/ (((m-r+ 0.5)/(M-m-R+r+1))))) * abs(((r)/(R)) -((m-r)/(M-R)))
        except ValueError:
            return 0
        return result

    def get_weighted_tags(self, genre_df_1, genre_df_2, model):
        tag_df_1 = genre_df_1.reset_index()
        tag_df_2 = genre_df_2.reset_index()
        unique_tags = list(tag_df_1.tag.unique()) + list(tag_df_2.tag.unique())
        temp_df_1 = genre_df_1.groupby(['movieid'])['tag'].apply(lambda x: ','.join(x)).reset_index()
        temp_df_2 = genre_df_2.groupby(['movieid'])['tag'].apply(lambda x: ','.join(x)).reset_index()
        genre_1_tag_counts = self.count_tags_in_movies(temp_df_1, genre_df_1.tag.unique(), model)
        temp_df_1 = genre_df_1.groupby(['movieid'])['tag'].apply(lambda x: ','.join(x)).reset_index()
        genre_2_tag_counts = self.count_tags_in_movies(pd.concat([temp_df_1, temp_df_2]).drop_duplicates(), unique_tags, model)
        weighted_tag_dict = {
        tag: self.compute_p1_weights(genre_1_tag_counts.get(tag, 0), genre_2_tag_counts.get(tag, 0),
                                     len(genre_df_1.index), len(pd.concat([genre_df_1, genre_df_2]).index))
        for tag in genre_1_tag_counts.keys()}
        return weighted_tag_dict

    def p1_differentiate_genre(self, genre_1, genre_2, model):
        genre_data_frame = self.get_genre_data()
        genre_1_data_frame = genre_data_frame[genre_data_frame.genre == genre_1]
        genre_2_data_frame = genre_data_frame[genre_data_frame.genre == genre_2]
        tag_dict = self.get_weighted_tags(genre_1_data_frame, genre_2_data_frame, model)
        tag_tuple = sorted(tag_dict.items(), key=operator.itemgetter(1), reverse=True)
        print({genre_1: dict(tag_tuple)})

if __name__ == "__main__":
    obj = GenreDifferentiator()
    #obj.differentiate_genre( "Children", "Animation",  "TFIDF")
    obj.p1_differentiate_genre("Children", "Animation", "P2_DIFF")