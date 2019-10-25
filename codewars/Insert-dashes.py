# Original problem can found at: https://www.codewars.com/kata/55960bbb182094bc4800007b
def insert_dash(num):
    num = str(num) 
    # 1 and 0 are considered True and False, i.e. if (1) is the same as if (True)
    # y % 2 is modulo 2 => gives 1 for odd numbers and 0 for even numbers
    return "".join([num[x] + "-" if x < len(num) - 1 and int(num[x]) % 2 and int(num[x + 1]) % 2 
    else num[x] for x in range(len(num))])
