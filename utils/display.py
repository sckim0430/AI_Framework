"""The print result implementation.
"""


def display_epoch(epoch, data_len, idx, metrics, data_time, batch_time):
    """The operation of print epoch result.

    Args:
        epoch (int): The epoch.
        data_len (int): The data length.
        idx (int): The index of loop.
        metrics (MetricMeter): The metrics to print.
        data_time (AverageMeter): The data loading time.
        batch_time (AverageMeter): The batch inference time.
    """
    entries = []
    entries += ['Epoch [{0:03d}]'.format(epoch)]
    entries += ['{:05d}/{:05d}'.format(idx, data_len)]
    entries += [metrics.__str__()]
    entries += [data_time.__str__()]
    entries += [batch_time.__str__()]
    print('  '.join(entries))


def display_test(data_len, metrics, data_time, batch_time):
    """The operation of print epoch result.

    Args:
        epoch (int): The epoch.
        data_len (int): The data length.
        idx (int): The index of loop.
        metrics (MetricMeter): The metrics to print.
        data_time (AverageMeter): The data loading time.
        batch_time (AverageMeter): The batch inference time.
    """
    entries = []
    entries += ['[Test Result of {:05d} Target]'.format(data_len)]
    entries += [metrics.__str__()]
    entries += ['Total Data Load Time {:.5f}'.format(data_time.sum)]
    entries += ['Total Data Infer Time {:.5f}'.format(batch_time.sum)]
    print('  '.join(entries))
