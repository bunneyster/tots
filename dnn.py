import constants
import numpy as np
import subprocess
import tensorflow as tf
import os

MODEL_DIR = './models/model'
TRAIN_DATA_PATH = './matches/csv/train_data.csv'
TEST_DATA_PATH = './matches/csv/test_data.csv'

TRAIN_EPOCHS = 40
TEST_EPOCHS = 2
BATCH_SIZE = 200
TRAIN_SIZE = 3677378
HIDDEN_UNITS = [23, 13, 3]

# Extract/return the features and labels from the CSV as a tuple.
#
# The features are in a dictionary that maps feature name to a list of Tensors,
# one Tensor for each
def input_fn(data_file, num_epochs, shuffle, batch_size):
    print(f"\nParsing file {data_file}")
    def parse_csv(row):
        columns = tf.decode_csv(row, record_defaults=constants.COLUMN_DEFAULTS)
        features = dict(zip(constants.COLUMN_NAMES, columns))
        labels = features.pop('outcome')
        return features, tf.equal(labels, 'won')

    dataset = tf.data.TextLineDataset(data_file)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=TRAIN_SIZE)

    # Set num_parallel_calls to number of CPUs.
    dataset = dataset.map(parse_csv, num_parallel_calls=4)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def hero_column_for(key):
    return tf.feature_column.categorical_column_with_vocabulary_list(
        key, constants.HEROES
    )

won_hero_cols = [hero_column_for(f"won_hero_{x + 1}") for x in range(5)]
won_hero_indicator_cols = [tf.feature_column.indicator_column(x) for x in won_hero_cols]

lost_hero_cols = [hero_column_for(f"lost_hero_{x + 1}") for x in range(5)]
lost_hero_indicator_cols = [tf.feature_column.indicator_column(x) for x in lost_hero_cols]

won_mmr_cols = [tf.feature_column.numeric_column(f"won_mmr_{x + 1}") for x in range(5)]
lost_mmr_cols = [tf.feature_column.numeric_column(f"lost_mmr_{x + 1}") for x in range(5)]

map_name = tf.feature_column.categorical_column_with_vocabulary_list(
    'map_name', constants.MAPS
)

duration = tf.feature_column.numeric_column('duration')

base_cols = won_hero_cols + lost_hero_cols + [map_name]

crossed_cols = []
for won_hero_column in won_hero_cols:
    for lost_hero_column in lost_hero_cols:
        crossed_cols.append(
            tf.feature_column.crossed_column(
                [won_hero_column, lost_hero_column],
                hash_bucket_size=1000
            )
        )

wide_cols = base_cols + crossed_cols
deep_cols = won_mmr_cols \
               + lost_mmr_cols \
               + [duration] \
               + won_hero_indicator_cols \
               + lost_hero_indicator_cols

for n in range(TRAIN_EPOCHS // TEST_EPOCHS):
    model = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=MODEL_DIR,
        linear_feature_columns=wide_cols,
        dnn_feature_columns=deep_cols,
        dnn_hidden_units=HIDDEN_UNITS
    )
    model.train(
        input_fn=lambda: input_fn(TRAIN_DATA_PATH, TRAIN_EPOCHS, True, BATCH_SIZE)
    )
    results = model.evaluate(
        input_fn=lambda: input_fn(TEST_DATA_PATH, TEST_EPOCHS, False, BATCH_SIZE)
    )

    print('Results at epoch', (n + 1) * 2)
    print('-' * 60)
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))
