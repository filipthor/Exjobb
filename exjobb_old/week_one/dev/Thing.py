class thing_class():

    def __init__(self,n):
        self.n = n


    def __call__(self):
        print("You called")


    def getN(self):
        return 2*self.n

    def __str__(self):
        return "Värde: " + str(self.n)
