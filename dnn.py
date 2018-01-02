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

# Return the next feature and label Tensors from given data file (.csv).
#
# Each element in the Dataset is a tuple of the form (features, label).
def input_fn(data_file, num_epochs, shuffle, batch_size):
    def parse_csv(row):
        columns = tf.decode_csv(row, record_defaults=constants.COLUMN_DEFAULTS)
        features = dict(zip(constants.COLUMN_NAMES, columns))
        labels = features.pop('outcome')
        return features, tf.equal(labels, 'won')

    dataset = tf.data.TextLineDataset(data_file)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=TRAIN_SIZE)

    dataset = dataset.map(parse_csv, num_parallel_calls=500)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def hero_column_for(key):
    return tf.feature_column.categorical_column_with_vocabulary_list(
        key, constants.HEROES
    )

won_hero_1 = hero_column_for('won_hero_1')
won_hero_2 = hero_column_for('won_hero_2')
won_hero_3 = hero_column_for('won_hero_3')
won_hero_4 = hero_column_for('won_hero_4')
won_hero_5 = hero_column_for('won_hero_5')
won_hero_columns = [won_hero_1, won_hero_2, won_hero_3, won_hero_4, won_hero_5]

lost_hero_1 = hero_column_for('lost_hero_1')
lost_hero_2 = hero_column_for('lost_hero_2')
lost_hero_3 = hero_column_for('lost_hero_3')
lost_hero_4 = hero_column_for('lost_hero_4')
lost_hero_5 = hero_column_for('lost_hero_5')
lost_hero_columns = [lost_hero_1, lost_hero_2, lost_hero_3, lost_hero_4, lost_hero_5]

won_mmr_1 = tf.feature_column.numeric_column('won_mmr_1')
won_mmr_2 = tf.feature_column.numeric_column('won_mmr_2')
won_mmr_3 = tf.feature_column.numeric_column('won_mmr_3')
won_mmr_4 = tf.feature_column.numeric_column('won_mmr_4')
won_mmr_5 = tf.feature_column.numeric_column('won_mmr_5')

lost_mmr_1 = tf.feature_column.numeric_column('lost_mmr_1')
lost_mmr_2 = tf.feature_column.numeric_column('lost_mmr_2')
lost_mmr_3 = tf.feature_column.numeric_column('lost_mmr_3')
lost_mmr_4 = tf.feature_column.numeric_column('lost_mmr_4')
lost_mmr_5 = tf.feature_column.numeric_column('lost_mmr_5')

map_name = tf.feature_column.categorical_column_with_vocabulary_list(
    'map_name', constants.MAPS
)

duration = tf.feature_column.numeric_column('duration')

base_columns = [
    won_hero_1, won_hero_2, won_hero_3, won_hero_4, won_hero_5,
    lost_hero_1, lost_hero_2, lost_hero_3, lost_hero_4, lost_hero_5,
    map_name
]
crossed_columns = []
for won_hero_column in won_hero_columns:
    for lost_hero_column in lost_hero_columns:
        crossed_columns.append(tf.feature_column.crossed_column([won_hero_column, lost_hero_column], hash_bucket_size=1000))

wide_columns = base_columns + crossed_columns
deep_columns = [
    won_mmr_1, won_mmr_2, won_mmr_3, won_mmr_4, won_mmr_5,
    lost_mmr_1, lost_mmr_2, lost_mmr_3, lost_mmr_4, lost_mmr_5,
    duration,
    tf.feature_column.indicator_column(won_hero_1),
    tf.feature_column.indicator_column(won_hero_2),
    tf.feature_column.indicator_column(won_hero_3),
    tf.feature_column.indicator_column(won_hero_4),
    tf.feature_column.indicator_column(won_hero_5),
    tf.feature_column.indicator_column(lost_hero_1),
    tf.feature_column.indicator_column(lost_hero_2),
    tf.feature_column.indicator_column(lost_hero_3),
    tf.feature_column.indicator_column(lost_hero_4),
    tf.feature_column.indicator_column(lost_hero_5),
    tf.feature_column.indicator_column(map_name)
]

for n in range(TRAIN_EPOCHS // TEST_EPOCHS):
    # run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'GPU': 0}))
    model = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=MODEL_DIR,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
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
