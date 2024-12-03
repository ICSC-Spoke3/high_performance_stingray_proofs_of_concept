import os
import subprocess as sp
import time
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from stingray import AveragedPowerspectrum, EventList, AveragedCrossspectrum
from stingray.fourier import positive_fft_bins
from stingray.gti import time_intervals_from_gtis, cross_two_gtis
from stingray.io import FITSTimeseriesReader
from stingray.loggingconfig import logger
from stingray.utils import histogram

# rng = np.random.default_rng(12345)


def _sum_arrays(array_generator, operation_on_data):
    total = operation_on_data(next(array_generator))
    for a in array_generator:
        total += operation_on_data(a)
    return total


def get_data_intervals(interval_idxs, timeseries, info=None, sample_time=None):

    if isinstance(timeseries, str):
        timeseries = FITSTimeseriesReader(timeseries, output_class=EventList)
    time_intervals = info["interval_times"][interval_idxs]
    if np.shape(time_intervals) == (2,):
        time_intervals = [time_intervals]

    # This creates a generator of event lists
    event_lists = timeseries.filter_at_time_intervals(time_intervals)

    for ev, t_int in zip(event_lists, time_intervals):
        nbin = int(np.rint((t_int[1] - t_int[0]) / sample_time))
        lc = histogram(ev.time, bins=nbin, range=(t_int[0], t_int[1]))
        yield lc


def single_rank_intervals(
    this_ranks_intervals, timeseries, sample_time=None, info=None
):

    t_int = info["interval_times"][0]
    nbin = int(np.rint((t_int[1] - t_int[0]) / sample_time))

    intv = info["interval_times"][0]
    segment_size = intv[1] - intv[0]
    if isinstance(timeseries, tuple):
        lc_iterable_1 = get_data_intervals(
            this_ranks_intervals, timeseries[0], info=info, sample_time=sample_time
        )
        lc_iterable_2 = get_data_intervals(
            this_ranks_intervals, timeseries[1], info=info, sample_time=sample_time
        )
        pds = AveragedCrossspectrum.from_lc_iterable(
            lc_iterable_1,
            lc_iterable_2,
            segment_size=segment_size,
            dt=sample_time,
            norm="leahy",
            silent=True,
        )
    else:
        lc_iterable = get_data_intervals(
            this_ranks_intervals, timeseries, info=info, sample_time=sample_time
        )
        pds = AveragedPowerspectrum.from_lc_iterable(
            lc_iterable,
            segment_size=segment_size,
            dt=sample_time,
            norm="leahy",
            silent=True,
        )
    return pds, nbin


def main_none(events, sample_time, segment_size):

    logger.info("Using standard Stingray processing")

    if isinstance(events, tuple):
        pds = AveragedCrossspectrum.from_events(
            events_or_tsreader(events[0]),
            events_or_tsreader(events[1]),
            dt=sample_time,
            segment_size=segment_size,
            norm="leahy",
            use_common_mean=False,
        )
    else:
        pds = AveragedPowerspectrum.from_events(
            events_or_tsreader(events),
            dt=sample_time,
            segment_size=segment_size,
            norm="leahy",
            use_common_mean=False,
        )

    return pds.freq, pds.power


def events_or_tsreader(events):
    if isinstance(events, str):
        return FITSTimeseriesReader(events, output_class=EventList)
    return events


def main_mpi(events, sample_time, segment_size):
    from mpi4py import MPI

    def data_lookup():
        # This will also contain the boundaries of the data
        # to be loaded
        if isinstance(events, tuple):
            gtis = cross_two_gtis(
                events_or_tsreader(events[0]).gti, events_or_tsreader(events[1]).gti
            )
        else:
            gtis = events_or_tsreader(events).gti

        start, stop = time_intervals_from_gtis(gtis, segment_size)
        interval_times = np.array(list(zip(start, stop)))
        return {
            "gtis": gtis,
            "interval_times": interval_times,
            "n_intervals": len(interval_times),
        }

    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()
    my_rank = world_comm.Get_rank()

    info = None
    if my_rank == 0:
        logger.debug(f"{my_rank}: Loading data")
        info = data_lookup()

    info = world_comm.bcast(info, root=0)

    if my_rank == 0:
        logger.debug(f"{my_rank}: Info:", info)

    total_n_intervals = info["n_intervals"]

    intervals_per_rank = total_n_intervals / world_size
    all_intervals = np.arange(total_n_intervals, dtype=int)

    this_ranks_intervals = all_intervals[
        (all_intervals >= my_rank * intervals_per_rank)
        & (all_intervals < (my_rank + 1) * intervals_per_rank)
    ]
    logger.debug(
        f"{my_rank}: Intervals {this_ranks_intervals[0] + 1} "
        f"to {this_ranks_intervals[-1] + 1}"
    )

    # data = get_data_intervals(this_ranks_intervals)
    result, data_size = single_rank_intervals(
        this_ranks_intervals, events, info=info, sample_time=sample_time
    )

    world_comm.Barrier()

    # NOW, the binary tree reduction
    totals = result.power * result.m
    previous_processors = np.arange(world_size, dtype=int)

    while 1:
        new_processors = previous_processors[::2]
        parner_processors = previous_processors[1::2]

        if my_rank == 0:
            logger.debug(f"{my_rank}: New processors: {new_processors}")
        if my_rank == 0:
            logger.debug(f"{my_rank}: Partners: {parner_processors}")
        if len(parner_processors) == 0:
            if my_rank == 0:
                logger.debug(f"{my_rank}: Done.")
            break
        world_comm.Barrier()

        for i, (sender, receiver) in enumerate(zip(parner_processors, new_processors)):
            if my_rank == 0:
                logger.debug(f"Loop {i + 1}: {sender}, {receiver}")
            tag = 10000 + 100 * sender + receiver
            if my_rank == receiver:
                data_from_partners = np.zeros_like(totals)
                logger.debug(f"{my_rank}: Receiving from {sender} with tag {tag}")
                # Might be good to use a non-blocking receive here
                world_comm.Recv(
                    data_from_partners,
                    source=sender,
                    tag=tag,
                )
                totals += data_from_partners
                logger.debug(f"{my_rank}: New data are now {totals}")
            elif my_rank == sender:
                # Only one partner for now. Might be tweaked differently. The rest should work with
                # any number of partners for a given processing rank

                logger.debug(f"{my_rank}: Sending to {receiver} with tag {tag}")
                world_comm.Send(totals, dest=receiver, tag=tag)
            else:
                logger.debug(f"{my_rank}: Doing nothing")

            world_comm.Barrier()
            previous_processors = new_processors

    world_comm.Barrier()

    assert len(new_processors) == 1
    if my_rank == new_processors[0]:
        logger.debug("Results")
        totals /= total_n_intervals

        freq = np.fft.fftfreq(data_size, d=sample_time)[positive_fft_bins(data_size)]

        return freq, totals
    return None, None


def main_multiprocessing(events, sample_time, segment_size, world_size=8):
    events_to_pass = events
    if isinstance(events, str):
        events = FITSTimeseriesReader(events, output_class=EventList)

    def data_lookup():
        # This will also contain the boundaries of the data
        # to be loaded
        if isinstance(events, tuple):
            gtis = cross_two_gtis(
                events_or_tsreader(events[0]).gti, events_or_tsreader(events[1]).gti
            )
        else:
            gtis = events_or_tsreader(events).gti

        start, stop = time_intervals_from_gtis(gtis, segment_size)
        interval_times = np.array(list(zip(start, stop)))
        return {
            "gtis": gtis,
            "interval_times": interval_times,
            "n_intervals": len(interval_times),
        }

    info = data_lookup()
    data_size = np.rint(segment_size / sample_time).astype(int)

    logger.debug("Info:", info)

    total_n_intervals = info["n_intervals"]

    # intervals_per_rank = total_n_intervals / world_size
    all_intervals = np.arange(total_n_intervals, dtype=int)

    intervals_per_rank = total_n_intervals / world_size
    this_ranks_intervals = []
    for my_rank in range(world_size):
        this_ranks_intervals.append(
            all_intervals[
                (all_intervals >= my_rank * intervals_per_rank)
                & (all_intervals < (my_rank + 1) * intervals_per_rank)
            ]
        )

    p = Pool(world_size)

    totals = 0

    for results, data_size in p.imap_unordered(
        partial(
            single_rank_intervals,
            timeseries=events_to_pass,
            info=info,
            sample_time=sample_time,
        ),
        this_ranks_intervals,
    ):
        totals += results.power * results.m
    logger.debug("Results")
    totals /= total_n_intervals

    freq = np.fft.fftfreq(data_size, d=sample_time)[positive_fft_bins(data_size)]

    return freq, totals


def simulate_data(length, ctrate, use_mpi=False, label=""):
    my_rank = 0
    if use_mpi:
        from mpi4py import MPI

        world_comm = MPI.COMM_WORLD
        my_rank = world_comm.Get_rank()

    label = f"_{label.lstrip('_')}"
    fname = f"fake_data_{ctrate:g}_cts_{length:g}_s{label}.evt"
    if my_rank == 0 and not os.path.exists(fname):
        logger.info("Simulating data")
        sp.check_call(
            f"HENfake -c {ctrate} --tstart 0 --tstop {length} --mjdref 56000 -o {fname}".split()
        )
    if use_mpi:
        world_comm.Barrier()
    return fname


def main_with_args(args=None):
    import argparse

    from hendrics.base import _add_default_args, check_negative_numbers_in_args

    description = (
        "Compute the power spectrum of an event list using different methods.\n"
        "To compare different methods, one might want to run the code multiple times, e.g.:\n"
        "python parallel_analysis_comparison.py filename.fits\n"
        "mpiexec -n 10 python parallel_analysis_comparison.py filename.fits --method mpi\n"
        "python parallel_analysis_comparison.py filename.fits --method multiprocessing --nproc 10\n"
    )
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("-c", "--count-rate", type=float, default=10)
    parser.add_argument("-l", "--length", type=float, default=12800)

    parser.add_argument(
        "-b",
        "--sample_time",
        type=float,
        default=1 / 8129 / 2,
        help="Light curve bin time; if negative, interpreted"
        + " as negative power of 2."
        + " Default: 2^-13, or keep input lc bin time"
        + " (whatever is larger)",
    )

    parser.add_argument(
        "-f",
        "--segment_size",
        type=float,
        default=128,
        help="Length of FFTs. Default: 16 s",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="leahy",
        help="Normalization to use"
        + " (Accepted: leahy and rms;"
        + ' Default: "leahy")',
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["mpi", "multiprocessing", "none"],
        default="none",
        help="Computation distribution method",
    )
    parser.add_argument(
        "--cross", action="store_true", default=False, help="Compute cross spectrum"
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=8,
        help="Number of processors to use",
    )
    parser.add_argument("--use-tsreader", action="store_true", default=False)

    _add_default_args(parser, ["loglevel", "debug"])

    args = check_negative_numbers_in_args(args)
    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = "DEBUG"

    logger.setLevel(args.loglevel)

    if args.count_rate * args.segment_size < 10:
        logger.warning(
            "The count rate is too low for the segment size. "
            "This will likely result in invalid results"
        )

    fname = simulate_data(args.length, args.count_rate, use_mpi=args.method == "mpi")
    if args.cross:
        fname2 = simulate_data(
            args.length, args.count_rate, use_mpi=args.method == "mpi", label="_2"
        )
        fname = (fname, fname2)

    if args.use_tsreader:
        print("Using TSReader")
        events = fname
    else:
        print("Using EventList")
        if isinstance(fname, tuple):
            events = tuple([EventList.read(f) for f in fname])
        else:
            events = EventList.read(fname)

    sample_time = args.sample_time
    segment_size = args.segment_size

    t0 = time.time()
    if args.method == "mpi":
        # This method needs to be run with mpiexec -n <nproc> python script.py
        freq, power = main_mpi(events, sample_time, segment_size)
    elif args.method == "multiprocessing":
        freq, power = main_multiprocessing(
            events, sample_time, segment_size, world_size=args.nproc
        )
    else:
        freq, power = main_none(events, sample_time, segment_size)
    if freq is None:
        return

    print(np.mean(power), "±", np.std(power))
    print("Elapsed time:", time.time() - t0)
    plt.figure()
    plt.plot(freq, power)
    if args.cross:
        plt.axhline(0)
    else:
        plt.axhline(2)
    plt.show()


if __name__ == "__main__":
    main_with_args()
