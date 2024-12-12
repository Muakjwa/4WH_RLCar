actions = {
    "S": [  
        "1,85",  
        "1,78",  
        "1,97", 
        "1,87"  
    ],
    "DR": [  
        "1,100",  
        "1,96",  
        "1,109", 
        "1,102"  
    ],
    "DR30": [  
        "1,115",  
        "1,111",  
        "1,124", 
        "1,117"  
    ],
    "DL": [  
        "1,70",  
        "1,66",  
        "1,79", 
        "1,72"  
    ],
    "DL30": [  
        "1,55",  
        "1,51",  
        "1,64", 
        "1,57"  
    ],
    "B9": [   
        "2,11.5", 
        "2,11.5", 
        "2,-11.5",  
        "2,-11.5" 
    ],
    "G9": [   
        "2,-3", 
        "2,-3", 
        "2,3",  
        "2,3" 
    ],
    "B18": [   
        "2,27", 
        "2,27", 
        "2,-27",  
        "2,-27" 
    ],
    "B36": [   
        "2,60", 
        "2,60", 
        "2,-60",  
        "2,-60" 
    ],
    "G36": [   
        "2,-60", 
        "2,-60", 
        "2,60",  
        "2,60" 
    ],
    "B300": [   
        "2,300", 
        "2,300", 
        "2,-300",  
        "2,-300" 
    ],
    "G": [   
        "2,-3", 
        "2,-3", 
        "2,3",  
        "2,3" 
    ],
    "G300": [   
        "2,-300", 
        "2,-300", 
        "2,300",  
        "2,300" 
    ],
    "T": [  
        "1,43",  
        "1,138",  
        "1,139", 
        "1,45"  
    ],
    "G3": [   
        "3,-3", 
        "3,-3", 
        "3,3",  
        "3,3" 
    ],
    
    # 8.5 original
    
    "TR": [   
        "3,8.5", 
        "3,8.5", 
        "3,8.5",  
        "3,8.5" 
    ],
    "TR100": [   
        "3,100", 
        "3,100", 
        "3,100",  
        "3,100" 
    ],
    "TL": [   
        "3,-8.5", 
        "3,-8.5", 
        "3,-8.5",  
        "3,-8.5" 
    ],
    "TL100": [   
        "3,-100", 
        "3,-100", 
        "3,-100",  
        "3,-100" 
    ],
}

def action_decision(action):
    mode = ''; move = 0
    if action == 0:
        mode = 'G'
        move = 9 
    if action == 1:
        mode = 'DR'
        move = 9 
    if action == 2:
        mode = 'DL'
        move = 9 
    if action == 3:
        mode = 'TR'
        move = 9 
    if action == 4:
        mode = 'TL'
        move = 9 
    if action == 5:
        mode = ''
        move = 9 
    if action == 6:
        mode = ''
        move = 9
    if action == 7:
        mode = ''
        move = 9
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
