def format_time(seconds, decimal_places=2):
    """
    Convert a time in seconds to a human-readable string with appropriate units,
    including SI units in parentheses for all time scales.
    
    Args:
    seconds (float): Time in seconds

    Returns:
    str: Formatted time string with appropriate units and SI units in parentheses
    """
    if seconds == 0:
        return "0 seconds (0 s)"

    units = [
        (60 * 60 * 24 * 365.25 * 1000000000000, "trillion years"),
        (60 * 60 * 24 * 365.25 * 1000000000, "billion years"),
        (60 * 60 * 24 * 365.25 * 1000000, "million years"),
        (60 * 60 * 24 * 365.25 * 1000, "thousand years"),
        (60 * 60 * 24 * 365.25 * 100, "century"),
        (60 * 60 * 24 * 365.25 * 10, "decade"),
        (60 * 60 * 24 * 365.25, "year"),
        (60 * 60 * 24 * 30.44, "month"),
        (60 * 60 * 24 * 7, "week"),
        (60 * 60 * 24, "day"),
        (60 * 60, "hour"),
        (60, "minute"),
        (1, "second"),
        (1e-3, "millisecond"),
        (1e-6, "microsecond"),
        (1e-9, "nanosecond"),
        (1e-12, "picosecond"),
        (1e-15, "femtosecond"),
        (1e-18, "attosecond"),
        (1e-21, "zeptosecond"),
        (1e-24, "yoctosecond")
    ]

    si_prefixes = [
        (1e24, "Y"),
        (1e21, "Z"),
        (1e18, "E"),
        (1e15, "P"),
        (1e12, "T"),
        (1e9, "G"),
        (1e6, "M"),
        (1e3, "k"),
        (1, ""),
        (1e-3, "m"),
        (1e-6, "μ"),
        (1e-9, "n"),
        (1e-12, "p"),
        (1e-15, "f"),
        (1e-18, "a"),
        (1e-21, "z"),
        (1e-24, "y")
    ]

    abs_seconds = abs(seconds)
    
    for unit_value, unit_name in units:
        if abs_seconds >= unit_value or unit_value == 1e-24:
            value = seconds / unit_value
            if abs(value) < 1 and unit_value > 1:
                continue
            rounded_value = round(value, decimal_places)
            if rounded_value.is_integer():
                rounded_value = int(rounded_value)
            # if unit_name.endswith('s') and abs(rounded_value) == 1:
            #     unit_name = unit_name[:-1]  # Remove 's' for singular units
            if not unit_name.endswith('s') and abs(rounded_value) != 1:
                if unit_name.endswith('y'):
                    unit_name = unit_name[:-1] + 'ies'  # Change 'y' to 'ies' for plural units
                else:
                    unit_name += 's'  # Add 's' for plural units
            
            formatted_time = f"{rounded_value} {unit_name}"
            
            # Add SI units in parentheses for all time scales
            for si_value, si_prefix in si_prefixes:
                if abs_seconds >= si_value or (abs_seconds < 1 and si_value == 1e-24):
                    si_rounded = round(seconds / si_value, decimal_places)
                    if si_rounded.is_integer():
                        si_rounded = int(si_rounded)
                    formatted_time += f" ({si_rounded} {si_prefix}s)"
                    break
            
            return formatted_time

if __name__ == "__main__":
    # Example usage
    def print_formatted_time(seconds):
        print(f"{seconds} seconds = {format_time(seconds)}")

    # Test cases
    print_formatted_time(1e-25)        # 10 yoctoseconds
    print_formatted_time(1e-22)        # 100 zeptoseconds
    print_formatted_time(1e-19)        # 100 attoseconds
    print_formatted_time(1e-16)        # 100 femtoseconds
    print_formatted_time(0.000000001)  # 1 nanosecond
    print_formatted_time(0.000001)     # 1 microsecond
    print_formatted_time(0.001)        # 1 millisecond
    print_formatted_time(1)            # 1 second
    print_formatted_time(30)           # 30 seconds
    print_formatted_time(60)           # 1 minute
    print_formatted_time(3600)         # 1 hour
    print_formatted_time(3900)         # 1 hour
    print_formatted_time(86400)        # 1 day
    print_formatted_time(604800)       # 1 week
    print_formatted_time(2592000+86400)      # 1 month (approx.)
    print_formatted_time(31536000+86400)     # 1 year (approx.)
    print_formatted_time(315360000+86400)    # 1 decade (approx.)
    print_formatted_time(3153600000+86400)   # 1 century (approx.)
    print_formatted_time(31536000000+86400)  # 1 thousand years (approx.)
    print_formatted_time(3.1536e13)    # 1 million years (approx.)
    print_formatted_time(3.1536e16)    # 1 billion years (approx.)
    print_formatted_time(-3600)        # Negative time (1 hour ago)