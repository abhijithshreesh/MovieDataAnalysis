import os
import logging
import pandas as pd
from sklearn.feature_selection.from_model import _LearntSelectorMixin

from config_parser import ParseConfig
from data_extractor import ExtractData
import sklearn
logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)
conf = ParseConfig()

class GenreDifferentiator(object):

    def __init__(self):
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")
        self.e_d = ExtractData(self.data_set_loc)

    def get_movie_genre_data(self):
        log.info("Extracting data from mlmovies.csv")
        m_a = self.e_d.data_extractor("mlmovies.csv")
        return m_a.reset_index()

    d

    def classify_movies_based_on_genre(self):
        data_frame = self.get_movie_genre_data()
        genre_data_frame = data_frame['genres'].str.split('|', expand=True).stack()
        genre_data_frame.name = "genre"
        genre_data_frame.index = genre_data_frame.index.droplevel(-1)
        genre_data_frame = genre_data_frame.reset_index()
        data_frame = data_frame.drop("genres", axis=1)
        data_frame = data_frame.reset_index()
        data_frame = genre_data_frame.merge(data_frame, how="left", on="index")
        data_frame = data_frame[["movieid","moviename", "genre"]]
        groupby_genre = data_frame.groupby("genre")
        for k, v in groupby_genre:
            print(v)

if __name__ == "__main__":
    obj = GenreDifferentiator()
    obj.classify_movies_based_on_genre()