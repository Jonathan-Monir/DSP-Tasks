file_path = "signal1.txt"

def clean_text(file_path):
    signals = list()
    with open(file_path, 'r') as file:
        for line in file:
            signals.append(line.strip())
            
    x = [item.split(' ')[0] for item in signals[3:]]
    y = [item.split(' ')[1] for item in signals[3:]]
    z = 0
    if signals[0] == '1':
        z = [item.split(' ')[2] for item in signals[3:]]
    return x, y, z
        
    
clean_text(file_path)