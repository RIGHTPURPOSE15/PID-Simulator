# config.py
fixed = 0.1
class config:
    def __init__(self, low, high, increment):
        """
        Initialize with default low/high values and increment for generating parameter values.
        These values are used by the individual generator functions.
        """
        self.low = low
        self.high = high
        self.increment = increment

    def Kp(self, fixed_ki=fixed, fixed_kd=fixed):
        """
        Generate a list of configurations with Kp varying,
        and Ki and Kd fixed to the provided values.
        """
        mods = []
        kp = self.low
        while kp <= self.high:
            mods.append({'kp': float(kp), 'ki': fixed_ki, 'kd': fixed_kd})
            kp += self.increment
        return mods

    def Ki(self, fixed_kp = fixed, fixed_kd=fixed):
        """
        Generate a list of configurations with Ki varying,
        and Kp and Kd fixed.
        """
        mods = []
        ki = self.low
        while ki <= self.high:
            mods.append({'kp': fixed_kp, 'ki': float(ki), 'kd': fixed_kd})
            ki += self.increment
        return mods

    def Kd(self, fixed_kp=fixed, fixed_ki=0.1):
        """
        Generate a list of configurations with Kd varying,
        and Kp and Ki fixed.
        """
        mods = []
        kd = self.low
        while kd <= self.high:
            mods.append({'kp': fixed_kp, 'ki': fixed_ki, 'kd': float(kd)})
            kd += self.increment
        return mods

    def PID(self, kp_range=None, ki_range=None, kd_range=None):
        """
        Generate a list of all combinations of PID parameters.
        Each range should be a tuple (low, high, increment). If a range is not provided,
        the defaults from the constructor will be used.
        """
        # Use provided ranges or defaults.
        kp_low, kp_high, kp_inc = kp_range if kp_range else (self.low, self.high, self.increment)
        ki_low, ki_high, ki_inc = ki_range if ki_range else (self.low, self.high, self.increment)
        kd_low, kd_high, kd_inc = kd_range if kd_range else (self.low, self.high, self.increment)

        mods = []
        kp = kp_low
        while kp <= kp_high:
            ki = ki_low
            while ki <= ki_high:
                kd = kd_low
                while kd <= kd_high:
                    mods.append({'kp': float(kp), 'ki': float(ki), 'kd': float(kd)})
                    kd += kd_inc
                ki += ki_inc
            kp += kp_inc
        return mods
