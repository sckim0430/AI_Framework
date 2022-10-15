"""The print result implementation.
"""
def display(epoch, data_len, idx, metrics, data_time, batch_time):
    """The operation of print result.

    Args:
        epoch (int): The epoch.
        data_len (int): The data length.
        idx (int): The index of train loop.
        metrics (MetricMeter): The metrics to print.
        data_time (AverageMeter): The data loading time.
        batch_time (AverageMeter): The batch train time.
    """
    entries = []
    entries += ['Epoch [{0:03d}] {}/{}:'.format(epoch,idx,data_len)]
    entries += [metrics.__str__()]
    entries += [data_time.__str__()]
    entries += [batch_time.__str__()]
    print('\t'.join(entries))
