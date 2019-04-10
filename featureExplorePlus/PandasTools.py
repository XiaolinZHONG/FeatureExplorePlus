import pandas as pd


def identify_missing(data):
    # Calculate the fraction of missing in each column
    missing_series = data.isnull().sum() / data.shape[0]
    missing_stats = pd.DataFrame(missing_series).rename(columns={'index': 'feature', 0: 'missing_fraction'})

    # Sort with highest number of missing values on top
    missing_stats = missing_stats.sort_values('missing_fraction', ascending=False)

    if ~missing_stats.empty:
        print('The missing fraction of columns:\n')
        with pd.option_context('display.max_rows', None):
            print(missing_stats[missing_stats['missing_fraction'] > 0.0])


def identify_single_unique(data):
    # Calculate the unique counts in each column
    unique_counts = data.nunique()
    unique_stats = pd.DataFrame(unique_counts).rename(columns={'index': 'feature', 0: 'nunique'})
    unique_stats = unique_stats.sort_values('nunique', ascending=True)

    # # Find the columns with only one unique count
    # record_single_unique = pd.DataFrame(unique_counts[unique_counts == 1]).reset_index().rename(
    #     columns={'index': 'feature',
    #              0: 'nunique'})
    if ~unique_stats.empty:
        print('Follows are the unique cloumns:\n')
        with pd.option_context('display.max_rows', None):
            print(unique_stats[unique_stats['nunique'] <= 1])


class PandasTools(object):

    def __init__(self, data_path, sep=",", nan_values=None, time_column=True, nrows=None, show_detial=True):
        self.data_path = data_path
        self.nan_values = nan_values
        self.time_column = time_column
        self.sep = sep
        self.show_detial = show_detial
        self.nrows = nrows

    def read_data(self, ):
        na_values = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', 'N/A',
                     'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null', '\\n', '\\N', "-900", "-1"]
        na_values = na_values.append(self.nan_values)
        reader = pd.read_csv(self.data_path, sep=self.sep, iterator=True, parse_dates=self.time_column,
                             na_values=na_values, nrows=self.nrows, infer_datetime_format=True,
                             error_bad_lines=False, engine='c')
        chunks = []
        loop = True
        i = 1
        while loop:
            try:
                chunk = reader.get_chunk(3000000000)
                chunks.append(chunk)
                print("Reading Data Chunks ", i)
                i += 1
            except StopIteration:
                loop = False
                print('Iteration is stopped.')
        data = pd.concat(chunks, ignore_index=True)
        print("Read Data Done")
        print('--' * 10)

        df0 = data.select_dtypes(include=["int"]).apply(pd.to_numeric, downcast="unsigned")
        df1 = data.select_dtypes(include=["float"]).apply(pd.to_numeric, downcast="float")
        df2 = data.select_dtypes(exclude=["int", "float"])
        data = pd.concat([df2, df1, df0], axis=1)
        print("The Shape of data:", data.shape)
        print('--' * 10)
        if self.show_detial:
            print(data.info(verbose=False))
            print('--' * 10)
            identify_missing(data)
            print('--' * 10)
            identify_single_unique(data)
            print('--' * 10)
            # with pd.option_context('display.max_columns', None):
            #     print('The Head 3 data:\n')
            #     print(data.head(3))
            #     print('\n')
        return data
