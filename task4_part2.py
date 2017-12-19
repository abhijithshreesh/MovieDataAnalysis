import logging
from config_parser import ParseConfig
from task_2 import GenreTag
import pandas as pd
import operator
from collections import Counter
import math
logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)
conf = ParseConfig()

class GenreDifferentiator(GenreTag):

    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")

    def count_tags_in_movies(self, data_frame, unique_tags):
        counter = {tag: 0 for tag in unique_tags}
        data_frame.tag = pd.Series([set(tags.split(',')) for tags in data_frame.tag], index=data_frame.index)
        for tag_list in data_frame.tag:
            for tag in tag_list:
                counter[tag] += 1
        return dict(counter)

    def compute_p1_weights(self, r, m , R, M):
        if r == m:
            return 0
        try:
            result = math.log(abs(((r)/(R-r))/ ((m-r)/(M-m-R+r)))) * abs(((r)/(R)) -((m-r)/(M-R)))
        except ValueError:
            return 0
        return result

    def get_weighted_tags(self, genre_df_1, genre_df_2):
        tag_df_1 = genre_df_1.reset_index()
        tag_df_2 = genre_df_2.reset_index()
        unique_tags = list(tag_df_1.tag.unique()) + list(tag_df_2.tag.unique())
        temp_df_1 = genre_df_1.groupby(['movieid'])['tag'].apply(lambda x: ','.join(x)).reset_index()
        temp_df_2 = genre_df_2.groupby(['movieid'])['tag'].apply(lambda x: ','.join(x)).reset_index()
        genre_1_tag_counts = self.count_tags_in_movies(temp_df_1, genre_df_1.tag.unique())
        temp_df_1 = genre_df_1.groupby(['movieid'])['tag'].apply(lambda x: ','.join(x)).reset_index()
        genre_2_tag_counts = self.count_tags_in_movies(pd.concat([temp_df_1, temp_df_2]).drop_duplicates(), unique_tags)
        weighted_tag_dict = {tag: self.compute_p1_weights(genre_1_tag_counts.get(tag, 0), genre_2_tag_counts.get(tag, 0),
                                                          len(genre_df_1.index), len(pd.concat([genre_df_1, genre_df_2]).index))
                                                    for tag in genre_1_tag_counts.keys()}
        return weighted_tag_dict

    def p1_differentiate_genre(self, genre_1, genre_2):
        genre_data_frame = self.get_genre_data()
        genre_1_data_frame = genre_data_frame[genre_data_frame.genre == genre_1]
        genre_2_data_frame = genre_data_frame[genre_data_frame.genre == genre_2]
        tag_dict = self.get_weighted_tags(genre_1_data_frame, genre_2_data_frame)
        tag_tuple = sorted(tag_dict.items(), key=operator.itemgetter(1), reverse=True)
        print({genre_1: dict(tag_tuple)})

if __name__ == "__main__":
    obj = GenreDifferentiator()
    obj.p1_differentiate_genre("Animation", "Children")