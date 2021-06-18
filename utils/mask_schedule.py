class RewarmSchedule:
    def __init__(self,start_epoch,end_epoch,end_rate,gamma):
        self.starte = start_epoch
        self.ende = end_epoch
        self.endr = end_rate
        self.g = gamma
        self.mide = (start_epoch + end_epoch)//2
    def __call__(self, epoch):
        if epoch < self.starte:
            return 0
        if epoch >= self.starte and epoch < self.mide:
            return self.endr * (self.g ** (self.mide-epoch))
        if epoch >= self.mide and epoch < self.ende:
            return self.endr * (self.g ** (self.ende - epoch))
        return self.endr


class WarmSchedule:
    def __init__(self,start_epoch,end_epoch,end_rate,gamma):
        self.starte = start_epoch
        self.ende = end_epoch
        self.endr = end_rate
        self.g = gamma
    def __call__(self, epoch):
        if epoch < self.starte:
            return 0
        if epoch >= self.starte and epoch < self.ende:
            return self.endr * (self.g ** (self.ende-epoch))
        return self.endr

class WarmScheduleLinear:
    def __init__(self,start_epoch,end_epoch,end_rate,gamma):
        self.starte = start_epoch
        self.ende = end_epoch
        self.endr = end_rate
        self.g = gamma
    def __call__(self, epoch):
        if epoch < self.starte:
            return 0
        if epoch >= self.starte and epoch < self.ende:
            return self.endr * (epoch-self.starte)/float(self.ende-self.starte)
        return self.endr


class MileStone:
    def __init__(self,miles_stones,values):
        assert len(miles_stones) == len(values)-1
        self.milestone = miles_stones
        self.values = values

    def __call__(self,epoch):
        for i,st in enumerate(self.milestone):
            if epoch <= st:
                return self.values[i]
        return self.values[-1]