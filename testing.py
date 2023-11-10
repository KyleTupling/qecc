import random # Find alternative for better randomness?

CYCLES = 1000 # How many times to repeat the encode->error->decode process

initialBitstring = "1011"

def strDifference(_str1, _str2):
    """Calculates how many characters two strings differ by
    
    Parameters:
        _str1 (str): First string
        _str2 (str): Second string

    Returns:
        int: How many characters the two strings differ by
    """
    count = 0
    for x, y in zip(_str1, _str2):
        count += x != y
    return count

def induceBitFlips(_p, _str):
    """Probabilistically induces bit-flips on a bitstring
    
    Parameters:
        _p (float): Probability of a bit flip occuring
        _str (str): The initial bitstring

    Returns:
        str: The bitstring with bit-flips induced
    """
    newStr = ""
    for i in range(len(_str)):
        if random.random() <= _p:
            newStr += str(int(not int(_str[i])))
        else:
            newStr += _str[i]
    return newStr

def encodeRepitition(_r, _str):
    """Encodes a bitstring with repitition
    
    Parameters:
        _r (int): How many times each bit should be repeated
        _str (str): The initial bitstring

    Returns:
        str: The repitition encoded bitstring
    """
    encodedStr = ""
    for i in range(len(_str)):
        for j in range(_r):
            encodedStr += _str[i]
    return encodedStr

def decodeRepitition_Majority(_r, _str):
    """Decodes a repitition encoded bitstring using majority vote
    
    Parameters:
        _r (int): How many repeats the bitstring has been encoded with
        _str (str): The repitition encoded bitstring

    Returns:
        str: The majority vote decoded bitstring
    """
    decodedStr = ""
    for i in range(0, len(_str), _r):
        zeros = [x for x in _str[i:i+3] if x == "0"]
        ones = [x for x in _str[i:i+3] if x == "1"]
        if len(zeros) > len(ones):
            decodedStr += "0"
        else:
            decodedStr += "1"
    return decodedStr

totalErrors = 0
for i in range(CYCLES): # Run process for given amount of cycles
    encodedStr = encodeRepitition(3, initialBitstring)
    errorStr = induceBitFlips(0.2, encodedStr)
    decodedStr = decodeRepitition_Majority(3, errorStr)
    totalErrors += strDifference(initialBitstring, decodedStr) # Accumulate amount of erroneous bits
totalBits = len(initialBitstring) * CYCLES
bitErrorRate = totalErrors / totalBits

print(f"Total Bits: {totalBits}  |  Total Errors: {totalErrors}  |  BER: {totalErrors/totalBits}")
