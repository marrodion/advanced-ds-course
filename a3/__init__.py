import logging
import logging.config
from pathlib import Path

f = Path(__file__)
log_file = f.parent / 'logging.cfg'

print("Configuring logger")
logging.config.fileConfig(log_file)