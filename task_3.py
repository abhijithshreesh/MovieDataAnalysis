import pandas as pd
import logging
from config_parser import ParseConfig
from task_2 import GenreTag

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)
conf = ParseConfig()

class UserTag(GenreTag):
    """
          Class to relate Users and tags, inherits the GenreTag to use the common weighing functons
    """
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")

    def merge_genre_tag(self, user, model):
        """
        Merges data from different csv files necessary to compute the tag weights for each user,
        assigns weights to timestamp.
        :param user:
        :param model:
        :return: returns a dictionary of Users to dictionary of tags and weights.
        """
        genome_tag = self.data_extractor.get_genome_tags_data()
        ml_tag = self.data_extractor.get_ml_tags_data()
        tag_data_frame = ml_tag.merge(genome_tag, how="left", left_on="tagid", right_on="tagId")
        data_frame_len = len(tag_data_frame.index)
        tag_data_frame["timestamp_weight"] = pd.Series(
            [(index + 1) / data_frame_len * 10 for index in tag_data_frame.index],
            index=tag_data_frame.index)
        tag_dict = self.combine_computed_weights(tag_data_frame[tag_data_frame["userid"] == user], model)
        print({user: tag_dict})

if __name__ == "__main__":
    obj = UserTag()
    obj.merge_genre_tag(user=109, model='TFIDF')

