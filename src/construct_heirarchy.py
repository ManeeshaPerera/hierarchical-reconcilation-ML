class ConstructHierarchy:
    def __init__(self, actual_data, number_of_levels):
        self.actual_data = actual_data
        self.ts_names = actual_data.columns
        self.number_of_levels = number_of_levels
        self.AGGREGATED_NAME = 'Aggregated'  # this is column is common across all datasets
        self.hierarchy_indexes = {}
        self.hierarchy_levels = {}
        self.bottom_level_start_index = None
        self.number_of_bottom_level_ts = 0
        self.construct_hierarchy_indexes()

    def construct_hierarchy_indexes(self):
        bottom_level_started = False
        for ts_index in range(0, len(self.ts_names)):
            ts_name = self.ts_names[ts_index]
            number_dashes = self.number_of_levels - 2  # level 1 and 2 dashes are 0 and from level 3 there exist a dash
            split_ts_name = ts_name.split("-", number_dashes)
            if split_ts_name[0] == self.AGGREGATED_NAME:
                self.hierarchy_levels[1] = [ts_name]
            else:
                level_index = len(split_ts_name) + 1  # levels start from 1
                if level_index in self.hierarchy_levels:
                    self.hierarchy_levels[level_index].append(ts_name)
                else:
                    self.hierarchy_levels[level_index] = [ts_name]

            if len(split_ts_name) > number_dashes:  # this is the bottom level
                self.number_of_bottom_level_ts += 1
                if not bottom_level_started:
                    self.bottom_level_start_index = ts_index
                    bottom_level_started = True
                self.hierarchy_indexes[0].append(ts_index)  # add to the top most level 1
                # now add to time series index to the other corresponding upper level nodes starting from level 2
                for hf_index in range(number_dashes):
                    node_name = "-".join(split_ts_name[0:hf_index + 1])  # find the upper level node name
                    node_index = self.ts_names.get_loc(node_name)  # get the node index
                    self.hierarchy_indexes[node_index].append(ts_index)
            else:
                self.hierarchy_indexes[ts_index] = []

    def get_hierarchy_indexes(self):
        return self.hierarchy_indexes

    def get_hierarchy_levels(self):
        return self.hierarchy_levels

    def get_bottom_level_ts_info(self):
        return self.number_of_bottom_level_ts, self.bottom_level_start_index

    def get_bottom_level_index(self):
        return self.bottom_level_start_index

    def get_bottom_level_no_ts(self):
        return self.number_of_bottom_level_ts

    def get_ts_names(self):
        return self.ts_names
