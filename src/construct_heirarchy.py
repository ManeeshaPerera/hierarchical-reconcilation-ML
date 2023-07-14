class ConstructHierarchy:
    def __init__(self, actual_data, number_of_levels):
        """
        Initialise the class parameters
        :param actual_data: pandas dataframe with columns for each time series in the hierarchy.
        Rows reflect the values and column names are time series names. The level 1 time series name is "Aggregated",
        from level 2 each column name will have dashes "-" to indicate the path of the hierarchy
        (e.g. A-B-C means time series C is connected to top level time series B and B is connected to top level timeseries A)
        :type actual_data: pandas dataframe
        :param number_of_levels: Number of levels in the hierarchy
        :type number_of_levels: integer
        """
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
        """
        This function is used to identify the connection between the leaf time series and other levels.
        E.g. consider a timeseries hierarchy with 3 levels and a structure A->B, C, B-> D, E, C-> F, G
        index values for these time series in the actual_data dataframe will be as follow A = 0, B=1, C=2, D=3, E=4, F=5, G=6
        hierarchy_indexes will be computed as follows: hierarchy_indexes = {0:[3, 4, 5, 6], 1: [3, 4], 2: [5, 6]}
        bottom_level_start_index = 3
        number_of_bottom_level_ts = 4
        hierarchy_levels = {1: ['Aggregated], 2: [B, C], 3: [B-D, B-E, C-F, C-G]
        """
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

    def get_non_bottom_level_ts_count(self):
        return len(self.ts_names) - self.number_of_bottom_level_ts
