import json

class B4RConfig:

    def __init__(
            self,
            path = "/home/jaesung/jaesung/study/bert4rec/config.json"
    ):
        self.path = path

        with open(self.path, 'r') as f:
            self.static_config = json.load(f)

        self.df = f"{self.static_config['data_folder']['home_dir']}/RAW_interactions.csv"