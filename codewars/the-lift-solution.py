# Original problem located at: https://www.codewars.com/kata/58905bfa1decb981da00009e
# Python 3: 884ms
# 'Collect' passengers travelling up till the max floor
# Then, we can just change direction upon the top floor and 'collect' passengers travelling down.
# Repeat if required.
# End once all passengers reach their destinations AND no more passengers in the lift.
class Dinglemouse(object):

    def __init__(self, queues, capacity):
        self.queues = [list(k) for k in queues]
        self.cap = capacity
        
    def theLift(self):
        q = self.queues
        direction = 1
        stopAtFloor = False
        passengers, result = [], [0]        
        
        while sum([len(x) for x in q]) > 0 or len(passengers) > 0: 
            start, end = 0, len(q) - 1
            if direction == -1:
                start, end = len(q) - 1, 0
            
            for x in range(start, end, direction):
                stopAtFloor = False
                
                # Let passengers whom reached their floor alight first
                if x in passengers:
                    stopAtFloor = True
                    # Others may alight elsewhere, have to remove 1 by 1
                    while x in passengers:
                        passengers.remove(x)
                
                # Add any new passengers, if within capacity
                if len(q[x]) > 0:
                    toRemove = []
                    for target in q[x]:
                        if target * direction > x * direction:
                            stopAtFloor = True
                            if len(passengers) < self.cap:
                                # Transfer passengers into lift
                                passengers.append(target)
                                toRemove.append(target)
                    
                    # Queue will reduce by passengers now in the lift
                    for s in toRemove:
                        q[x].remove(s)
                
                if stopAtFloor and result[-1] != x:
                    result.append(x)
            
            direction *= -1
            
        if len(result) > 1 and result[-1] != 0:    
            result.append(0)
        return result
