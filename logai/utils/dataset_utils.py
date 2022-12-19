from logai.dataloader.data_model import LogRecordObject
from logai.utils import constants
from sklearn.model_selection import train_test_split


def split_train_dev_test_for_anomaly_detection(
    logrecord, training_type, test_data_frac_pos_class, test_data_frac_neg_class
):
    id_label_df = logrecord.labels.join([logrecord.span_id])
    ids_pos = id_label_df[id_label_df[constants.LABELS] == 1][
        constants.SPAN_ID
    ].tolist()
    ids_neg = id_label_df[id_label_df[constants.LABELS] == 0][
        constants.SPAN_ID
    ].tolist()

    ids_pos = list(set(ids_pos))
    ids_neg = list(set(ids_neg))
    if training_type == "supervised":
        train_ids_pos, test_ids_pos = train_test_split(
            ids_pos, test_size=test_data_frac_pos_class
        )
        if len(train_ids_pos) * 0.1 > 1:
            train_ids_pos, dev_ids_pos = train_test_split(train_ids_pos, test_size=0.1)
        else:
            dev_ids_pos = []

    elif training_type == "unsupervised":
        test_ids_pos = ids_pos
        train_ids_pos = []
        dev_ids_pos = []

    else:
        raise Exception(
            "unknown training_type {} should be 'supervised' or 'unsupervised'".format(
                training_type
            )
        )
    train_ids_neg, test_ids_neg = train_test_split(
        ids_neg, test_size=test_data_frac_neg_class
    )
    if len(train_ids_neg) * 0.1 > 1:
        train_ids_neg, dev_ids_neg = train_test_split(train_ids_neg, test_size=0.1)
    else:
        dev_ids_neg = []
    train_ids = []
    train_ids.extend(train_ids_pos)
    train_ids.extend(train_ids_neg)

    dev_ids = []
    dev_ids.extend(dev_ids_pos)
    dev_ids.extend(dev_ids_neg)

    test_ids = []
    test_ids.extend(test_ids_pos)
    test_ids.extend(test_ids_neg)

    indices_train = list(
        logrecord.span_id.loc[
            logrecord.span_id[constants.SPAN_ID].isin(train_ids)
        ].index
    )
    indices_dev = list(
        logrecord.span_id.loc[
            logrecord.span_id[constants.SPAN_ID].isin(dev_ids)
        ].index
    )
    indices_test = list(
        logrecord.span_id.loc[
            logrecord.span_id[constants.SPAN_ID].isin(test_ids)
        ].index
    )

    logrecord_train = logrecord.select_by_index(indices_train)
    logrecord_dev = logrecord.select_by_index(indices_dev)
    logrecord_test = logrecord.select_by_index(indices_test)

    return logrecord_train, logrecord_dev, logrecord_test
