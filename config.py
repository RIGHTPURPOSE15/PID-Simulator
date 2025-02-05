class config:
    def __init__(self, range0, range1, increment):
        self.range0 = range0
        self.range1 = range1
        self.increment = increment
        self.k = None  # Currently unused but retained
        self.mod = []  # Initialize an empty list

    def Kp(self):
        self.k = 'Kp'
        self.mod = []  # Reset list before generating new values

        i = self.range0
        while i <= self.range1:
            self.mod.append({'kp': float(i), 'ki': 0.1, 'kd': 0.1})
            i += self.increment
        
        return self.mod