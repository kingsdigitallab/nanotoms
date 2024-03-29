import logging
import logging.config

__version__ = "__version__ = '__version__ = '0.2.1''"
__version_info__ = tuple(
    [
        int(num) if num.isdigit() else num
        for num in __version__.replace("-", ".", 1).split(".")
    ]
)

logging.config.fileConfig("nanotoms/../logging.conf")
