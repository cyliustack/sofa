class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    HINT ='\033[38;5;45m' 
    TEXT_PURPLE ='\033[38;5;104m' 
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    DATA = '\033[5;30;47m'
    TITLE = '\033[48;5;26m'
    C_NONE = '\033[0;00m'
    C_RED = '\033[1;31m'
    C_GREEN = '\033[1;32m'

def print_title(content):
    print('\n')
    print(bcolors.TITLE + content + bcolors.ENDC)

def print_error(content):
    print(bcolors.C_RED + "[ERROR] " + content + bcolors.ENDC)

def print_warning(cfg, content):
    if cfg.verbose:
        print(bcolors.WARNING + "[WARNING] " + content + bcolors.ENDC)

def print_info(cfg, content):
    if cfg.verbose:
        print(bcolors.OKGREEN + "[INFO] " + content + bcolors.ENDC)

def print_hint(content):
    print(bcolors.HINT + "[HINT] " + content + bcolors.ENDC)

def print_progress(cfg, content):
    if cfg.verbose:
        print(bcolors.TEXT_PURPLE + "[PROGRESS] " + content + bcolors.ENDC)

def print_main_progress(content):
    print(bcolors.TEXT_PURPLE + "[PROGRESS] " + content + bcolors.ENDC)

def highlight(content):
    return bcolors.OKGREEN + content + bcolors.ENDC

def print_data(content):
    print(bcolors.DATA)
    print(content)
    print(bcolors.ENDC)

# print_format_table() refers to
# https://stackoverflow.com/posts/21786287/revisions


def print_format_table():
    """
    prints table of formatted text format options
    """
    for style in range(8):
        for fg in range(30, 38):
            s1 = ''
            for bg in range(40, 48):
                format = ';'.join([str(style), str(fg), str(bg)])
                s1 += '\x1b[%sm %s \x1b[0m' % (format, format)
            print(s1)
        print('\n')
