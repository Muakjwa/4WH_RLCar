actions = {
    "S": [  # ??
        "1,85",  # Teensy 1: ??? 0?
        "1,78",  # Teensy 2: ??? 0?
        "1,97",  # Teensy 3: ??? 0?
        "1,87"   # Teensy 4: ??? 0?
    ],
    "DR": [  # ??
        "1,100",  # Teensy 1: ??? 0?
        "1,96",  # Teensy 2: ??? 0?
        "1,109",  # Teensy 3: ??? 0?
        "1,102"   # Teensy 4: ??? 0?
    ],
    "DR30": [  # ??
        "1,115",  # Teensy 1: ??? 0?
        "1,111",  # Teensy 2: ??? 0?
        "1,124",  # Teensy 3: ??? 0?
        "1,117"   # Teensy 4: ??? 0?
    ],
    "DL": [  # ??
        "1,70",  # Teensy 1: ??? 0?
        "1,66",  # Teensy 2: ??? 0?
        "1,79",  # Teensy 3: ??? 0?
        "1,72"   # Teensy 4: ??? 0?
    ],
    "DL30": [  # ??
        "1,55",  # Teensy 1: ??? 0?
        "1,51",  # Teensy 2: ??? 0?
        "1,64",  # Teensy 3: ??? 0?
        "1,57"   # Teensy 4: ??? 0?
    ],
    "B9": [  # ?? ??
        "2,11.5",  # Teensy 1: 100cm ??
        "2,11.5",  # Teensy 2: 100cm ??
        "2,-11.5",  # Teensy 3: 100cm ??
        "2,-11.5"   # Teensy 4: 100cm ??
    ],
    "G9": [  # ?? ??
        "2,-3",  # Teensy 1: 100cm ??
        "2,-3",  # Teensy 2: 100cm ??
        "2,3",  # Teensy 3: 100cm ??
        "2,3"   # Teensy 4: 100cm ??
    ],
    "B18": [  # ?? ??
        "2,27",  # Teensy 1: 100cm ??
        "2,27",  # Teensy 2: 100cm ??
        "2,-27",  # Teensy 3: 100cm ??
        "2,-27"   # Teensy 4: 100cm ??
    ],
    "B36": [  # ?? ??
        "2,60",  # Teensy 1: 100cm ??
        "2,60",  # Teensy 2: 100cm ??
        "2,-60",  # Teensy 3: 100cm ??
        "2,-60"   # Teensy 4: 100cm ??
    ],
    "G36": [  # ?? ??
        "2,-60",  # Teensy 1: 100cm ??
        "2,-60",  # Teensy 2: 100cm ??
        "2,60",  # Teensy 3: 100cm ??
        "2,60"   # Teensy 4: 100cm ??
    ],
    "B300": [  # ?? ??
        "2,300",  # Teensy 1: 100cm ??
        "2,300",  # Teensy 2: 100cm ??
        "2,-300",  # Teensy 3: 100cm ??
        "2,-300"   # Teensy 4: 100cm ??
    ],
    "G": [  # ?? ??
        "2,-3",  # Teensy 1: 100cm ??
        "2,-3",  # Teensy 2: 100cm ??
        "2,3",  # Teensy 3: 100cm ??
        "2,3"   # Teensy 4: 100cm ??
    ],
    "G300": [  # ?? ??
        "2,-300",  # Teensy 1: 100cm ??
        "2,-300",  # Teensy 2: 100cm ??
        "2,300",  # Teensy 3: 100cm ??
        "2,300"   # Teensy 4: 100cm ??
    ],
    "T": [  # ??
        "1,43",  # Teensy 1: ??? 0?
        "1,138",  # Teensy 2: ??? 0?
        "1,139",  # Teensy 3: ??? 0?
        "1,45"   # Teensy 4: ??? 0?
    ],
    "G3": [  # ?? ??
        "3,-3",  # Teensy 1: 100cm ??
        "3,-3",  # Teensy 2: 100cm ??
        "3,3",  # Teensy 3: 100cm ??
        "3,3"   # Teensy 4: 100cm ??
    ],
    
    # 8.5 original
    
    "TR": [  # ?? ??
        "3,8.5",  # Teensy 1: 100cm ??
        "3,8.5",  # Teensy 2: 100cm ??
        "3,8.5",  # Teensy 3: 100cm ??
        "3,8.5"   # Teensy 4: 100cm ??
    ],
    "TR100": [  # ?? ??
        "3,100",  # Teensy 1: 100cm ??
        "3,100",  # Teensy 2: 100cm ??
        "3,100",  # Teensy 3: 100cm ??
        "3,100"   # Teensy 4: 100cm ??
    ],
    "TL": [  # ?? ??
        "3,-8.5",  # Teensy 1: 100cm ??
        "3,-8.5",  # Teensy 2: 100cm ??
        "3,-8.5",  # Teensy 3: 100cm ??
        "3,-8.5"   # Teensy 4: 100cm ??
    ],
    "TL100": [  # ?? ??
        "3,-100",  # Teensy 1: 100cm ??
        "3,-100",  # Teensy 2: 100cm ??
        "3,-100",  # Teensy 3: 100cm ??
        "3,-100"   # Teensy 4: 100cm ??
    ],
}

def action_decision(action):
    mode = ''; move = 0
    if action == 0:
        mode = 'G'
        move = 9 # 1/2??
    if action == 1:
        mode = 'DR'
        move = 9 # 1/2??
    if action == 2:
        mode = 'DL'
        move = 9 # 1/2??
    if action == 3:
        mode = 'TR'
        move = 9 # 1/2??
    if action == 4:
        mode = 'TL'
        move = 9 # 1/2??
    if action == 5:
        mode = ''
        move = 9 # 1/2??
    if action == 6:
        mode = ''
        move = 9 # 1??
    if action == 7:
        mode = ''
        move = 9 # 2??
    return mode, move

def set_posture(mode):
    if mode == 'G':
        return 'S'
    elif mode == 'DR':
        return 'DR'
    elif mode == 'DL':
        return 'DL'
    elif mode == 'TL':
        return 'TL'
    elif mode == 'TR':
        return 'TR'
    else:
        return ''
    
def set_distance(move):
    if move == 9:
        return 'G9'
    elif move == 18:
        return 'G18'
    elif move == 36:
        return 'G36'
    else:
        return ''
