'''
Custom log object to use across all files
'''
import logging

def make_logger(name, logFile, debugMode = False, silent = False):
    
    # Create a custom log
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    # Create handlers
    #setting level
    if debugMode:
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(logFile)
        c_handler.setLevel(logging.DEBUG)
        f_handler.setLevel(logging.DEBUG)
    elif silent:
        f_handler = logging.FileHandler(logFile)
        f_handler.setLevel(logging.INFO)
    else:
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(logFile)
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)        


    # Create formatters and add it to handlers
    f_format = logging.Formatter('%(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    # Add handlers to the log
    log.addHandler(f_handler)

    if not silent:
        c_format = logging.Formatter('%(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        log.addHandler(c_handler)

    return log