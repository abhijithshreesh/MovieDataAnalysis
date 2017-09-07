import os
import pandas as pd
import logging
from config_parser import ParseConfig
from data_extractor import ExtractData

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

conf = ParseConfig()

class ActorTag(object):

    def __init__(self):
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")
        self.e_d = ExtractData(self.data_set_loc)

    def get_movie_actor_data(self):
        log.info("Extracting data from movie-actor.csv")
        m_a = self.e_d.data_extractor("movie-actor.csv")
        return m_a.reset_index()

    def get_ml_tags_data(self):
        log.info("Extracting data from mltags.csv")
        a_t = self.e_d.data_extractor("mltags.csv")
        return a_t.reset_index()

    def get_genome_tags_data(self):
        log.info("Extracting data from genome-tags.csv")
        g_t = self.e_d.data_extractor("genome-tags.csv")
        return g_t.reset_index()

    def get_imdb_actor_info(self):
        log.info("Extracting data from imdb-actor-info.csv")
        actor_info = self.e_d.data_extractor("imdb-actor-info.csv")
        return actor_info.reset_index()

    def assign_weight(self, data_frame):
        data_frame = data_frame.drop(["userid", "movieid"], axis=1)
        data_frame = data_frame.fillna('')
        actor_tag_dict = {}
        groupby_actor_dict = data_frame.groupby("name")
        for actorid, tag_df in groupby_actor_dict:
            tag_df = tag_df.sort_values("timestamp")
            actor_tag_dict[actorid] = tag_df.tag.tolist()
        actor_tag_data_frame = pd.DataFrame.from_dict(list(actor_tag_dict.items()))
        return actor_tag_data_frame

    def merge_movie_actor_and_tag(self):
        mov_act = self.get_movie_actor_data()
        ml_tag = self.get_ml_tags_data()
        genome_tag = self.get_genome_tags_data()
        actor_info = self.get_imdb_actor_info()
        actor_movie_info = mov_act.merge(actor_info, how="left", on="actorid")
        tag_data_frame = ml_tag.merge(genome_tag, how="left", left_on="tagid", right_on="tagId")
        merged_data_frame = actor_movie_info.merge(tag_data_frame, how="left", on="movieid")
        actor_tag_data_frame = self.assign_weight(merged_data_frame)
        actor_tag_data_frame.columns = ['actor_name', 'tags']
        actor_tag_data_frame.to_csv(os.path.join(self.data_set_loc, "ACT_TAG.csv"))


if __name__ == "__main__":
    obj = ActorTag()
    obj.merge_movie_actor_and_tag()

