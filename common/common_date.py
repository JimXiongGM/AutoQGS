import datetime
import time


def date_range(start="2020-12-30", end="2021-01-03", step=1, format="%Y-%m-%d", exclude_dates=[]):
    """
    default input date format: 2021-01-03
    return: list of date items
    exclude_dates: list. item can be date list or data range. e.g. `2021-10-01` or `2021-10-01~2021-10-07`
    """
    # check the format
    ex_datas = []
    for item in exclude_dates:
        if len(item) == 10:
            ex_datas.append(item)
        elif len(item) == (2 * 10 + 1):
            tmp = date_range(start=item.split("~")[0], end=item.split("~")[1], step=1)
            ex_datas.extend(tmp)
        else:
            raise ValueError("err item: " + item)

    # begin
    strptime, strftime = datetime.datetime.strptime, datetime.datetime.strftime
    days = (strptime(end, format) - strptime(start, format)).days
    dates = [
        strftime(strptime(start, format) + datetime.timedelta(i), format) for i in range(0, days + 1, step)
    ]
    dates = [date for date in dates if date not in ex_datas]
    return dates


def time_convert(
    from_formats=[
        "%m/%d/%Y %H:%M:%S",
    ],
    to_format="%Y-%m-%d %H:%M:%S",
    in_time="2018-09-09 21:37:45",
    return_status=False,
):
    """
    convert time format
    multiple inputs --> one output
    usages:
        - %A full week name
        - %d decimal number [01,31] representing the day of the month.
        - %B full month name
        - %Y four-digit year
        - readable: strftime("%d %B %Y %H:%M:%S", time_array)
    """
    if not in_time:
        return None
    status = False
    for _format in from_formats:
        try:
            # timestamp
            time_array = time.strptime(in_time, _format)
            # convert to output format
            newtime = time.strftime(to_format, time_array)
            status = True
            break
        except ValueError:
            newtime = in_time
    return (newtime, status) if return_status else newtime
